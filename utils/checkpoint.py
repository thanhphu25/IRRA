# Checkpointer API based on the original IRRA/Facebook implementation.
"""Strict model restoration and explicit baseline-to-prompt warm starts."""
import logging
import os
import random
from collections import OrderedDict

import numpy as np
import torch


def strip_prefix_if_present(state_dict, prefix="module."):
    result = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[len(prefix):] if key.startswith(prefix) else key
        if new_key in result:
            raise RuntimeError(f"Duplicate checkpoint key after prefix removal: {new_key}")
        result[new_key] = value
    return result


def load_state_dict(model, loaded_state_dict, allow_missing_prompts=False):
    """Only explicit baseline warm-start may omit the entire spatial module."""
    target = model.module if hasattr(model, 'module') else model
    source = strip_prefix_if_present(loaded_state_dict)
    expected = target.state_dict()
    missing, unexpected = sorted(set(expected) - set(source)), sorted(set(source) - set(expected))
    mismatched = [f"{k}: checkpoint {tuple(source[k].shape)} != model {tuple(expected[k].shape)}"
                  for k in set(expected) & set(source) if source[k].shape != expected[k].shape]
    logging.getLogger("IRRA.checkpoint").info("Missing keys: %s; Unexpected keys: %s", missing, unexpected)
    allowed = (allow_missing_prompts and not any(k.startswith('spatial_prompt.') for k in source)
               and all(k.startswith('spatial_prompt.') for k in missing))
    if unexpected or mismatched or (missing and not allowed):
        raise RuntimeError(f"Incompatible checkpoint. Missing: {missing}; Unexpected: {unexpected}; "
                           f"Shape mismatches: {mismatched}")
    target.load_state_dict(source, strict=not bool(missing))
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def _rng_state():
    state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    name, keys, pos, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


class Checkpointer:
    def __init__(self, model, optimizer=None, scheduler=None, save_dir="", save_to_disk=None, logger=None):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.save_dir, self.save_to_disk = save_dir, save_to_disk
        self.logger = logger or logging.getLogger("IRRA.checkpoint")

    @property
    def target(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def save(self, name, **kwargs):
        if not self.save_dir or not self.save_to_disk:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        data = {"model": self.target.state_dict(), "format_version": 2,
                "model_config": dict(vars(self.target.args)) if hasattr(self.target, 'args') else {},
                "rng_state": _rng_state()}
        if self.optimizer is not None:
            data['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data['scheduler'] = self.scheduler.state_dict()
        data.update(kwargs)
        path = os.path.join(self.save_dir, name + '.pth')
        self.logger.info("Saving checkpoint to %s", path)
        torch.save(data, path)

    def _load_file(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Checkpoint must be an IRRA payload or tensor state_dict")
        if 'model' not in checkpoint:
            if not checkpoint or not all(torch.is_tensor(v) for v in checkpoint.values()):
                raise RuntimeError("Checkpoint has neither model weights nor a tensor state_dict")
            checkpoint = {"model": checkpoint}
        return checkpoint

    def _load_model(self, checkpoint, warm_start=False):
        source = strip_prefix_if_present(checkpoint['model'])
        saved = checkpoint.get('model_config', {})
        # Non-tensor options change the computation too: prevent silent mismatches.
        if any(k.startswith('spatial_prompt.') for k in source) and hasattr(self.target, 'args'):
            current = vars(self.target.args)
            relevant = [k for k in saved if k == 'lgsp_mode' or
                        k.startswith(('frequency_', 'local_')) and k != 'local_rank' and not k.endswith('_alpha_init')]
            differences = [k for k in relevant if saved[k] != current.get(k)]
            if differences:
                raise RuntimeError(f"Prompt configuration differs from checkpoint: {differences}")
        # A new prompt checkpoint with *all* prompt tensors removed is corrupt too.
        if saved.get('lgsp_mode', 'baseline') != 'baseline' and not any(k.startswith('spatial_prompt.') for k in source):
            raise RuntimeError("Checkpoint declares LGSP but all spatial_prompt keys are missing")
        return load_state_dict(self.model, source, allow_missing_prompts=warm_start)

    def load(self, f=None):
        """Strict model-only restore for evaluation of a trained model."""
        if not f:
            raise ValueError("A checkpoint path is required")
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        return {k: v for k, v in checkpoint.items() if k not in ('model', 'optimizer', 'scheduler', 'rng_state')}

    def warm_start(self, path):
        """Initialize weights, permitting only missing new prompts; never load optimizer/RNG."""
        checkpoint = self._load_file(path)
        report = self._load_model(checkpoint, warm_start=True)
        self.logger.info("Warm start %s; optimizer/scheduler remain fresh. %s", path, report)
        return report

    def resume(self, f=None):
        if not f:
            raise ValueError("A resume checkpoint path is required")
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        for key, component in (("optimizer", self.optimizer), ("scheduler", self.scheduler)):
            if component is not None:
                if key not in checkpoint:
                    raise RuntimeError(f"Cannot resume: missing {key} state")
                component.load_state_dict(checkpoint[key])
        if 'rng_state' in checkpoint:
            _restore_rng(checkpoint['rng_state'])
        return {k: v for k, v in checkpoint.items() if k not in ('model', 'optimizer', 'scheduler', 'rng_state')}
