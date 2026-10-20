"""Strict IRRA/SCFP checkpoints. Warm-start never restores optimizer or epoch."""
import logging
import json
import os
import random

import numpy as np
import torch


NEW_PREFIXES = ('base_model.visual.scfp_frequency.', 'scfp_slots.')
ARCHITECTURE_FIELDS = ('scfp_mode', 'scfp_num_slots', 'scfp_slot_heads',
                       'scfp_frequency_layer', 'scfp_frequency_num_bands',
                       'scfp_frequency_hidden_dim', 'scfp_frequency_temperature',
                       'scfp_frequency_variant', 'img_size', 'stride_size',
                       'text_length', 'vocab_size', 'cmt_depth', 'loss_names')


def checkpoint_mode(checkpoint):
    if 'scfp_mode' in checkpoint:
        return checkpoint['scfp_mode']
    if 'scfp_mode' in checkpoint.get('config', {}):
        return checkpoint['config']['scfp_mode']
    keys = [k.removeprefix('module.') for k in checkpoint['model']]
    frequency = any(k.startswith(NEW_PREFIXES[0]) for k in keys)
    slots = any(k.startswith(NEW_PREFIXES[1]) for k in keys)
    return 'v2' if frequency and slots else 'frequency' if frequency else 'slots' if slots else 'baseline'


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def strip_prefix_if_present(state_dict, prefix='module.'):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def load_state_dict(model, loaded_state_dict, except_keys=None, allow_new=False):
    if except_keys:
        raise ValueError('Selective silent checkpoint loading is not supported')
    model = unwrap(model)
    loaded = strip_prefix_if_present(loaded_state_dict)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(loaded))
    unexpected = sorted(set(loaded) - set(expected))
    mismatches = [f'{k}: {tuple(loaded[k].shape)} != {tuple(expected[k].shape)}'
                  for k in set(expected) & set(loaded) if loaded[k].shape != expected[k].shape]
    # Only an ORIGINAL IRRA state may omit newly introduced parameters.
    original_irra = not any(k.startswith(NEW_PREFIXES) for k in loaded)
    illegal = [k for k in missing if not (allow_new and original_irra and k.startswith(NEW_PREFIXES))]
    report = dict(missing_keys=missing, unexpected_keys=unexpected, incompatible_shapes=mismatches)
    logging.getLogger('IRRA.checkpoint').info('Checkpoint compatibility: %s', report)
    if illegal or unexpected or mismatches:
        raise RuntimeError(f'Incompatible checkpoint: {report}')
    model.load_state_dict(loaded, strict=not missing)
    return report


def rng_state():
    np_state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=(np_state[0], np_state[1].tolist(), *np_state[2:]),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


class Checkpointer:
    def __init__(self, model, optimizer=None, scheduler=None, save_dir='',
                 save_to_disk=None, logger=None):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.save_dir, self.save_to_disk = save_dir, save_to_disk
        self.logger = logger or logging.getLogger('IRRA.checkpoint')
        self.last_report = None

    def save(self, name, **kwargs):
        if not self.save_dir or not self.save_to_disk:
            return
        model = unwrap(self.model)
        data = dict(model=model.state_dict(), config=dict(vars(model.args)),
                    scfp_mode=getattr(model, 'scfp_mode', 'baseline'),
                    num_classes=getattr(model, 'num_classes', None), rng_state=rng_state())
        provenance_path = os.path.join(self.save_dir, 'environment.json')
        if os.path.isfile(provenance_path):
            with open(provenance_path) as stream:
                data['provenance'] = json.load(stream)
        if self.optimizer is not None:
            data['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data['scheduler'] = self.scheduler.state_dict()
        data.update(kwargs)
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f'{name}.pth')
        temporary = path + '.tmp'
        torch.save(data, temporary)
        os.replace(temporary, path)
        self.logger.info('Saved checkpoint %s', path)

    def _load_file(self, path):
        # Only load user-provided, trusted IRRA experiment checkpoints.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        return checkpoint if 'model' in checkpoint else {'model': checkpoint}

    def _validate_metadata(self, checkpoint):
        model = unwrap(self.model)
        target = vars(model.args)
        saved = checkpoint.get('config', {})
        mode = checkpoint_mode(checkpoint)
        if mode != getattr(model, 'scfp_mode', 'baseline'):
            raise RuntimeError(f'Checkpoint mode {mode} differs from model {model.scfp_mode}')
        for key in ARCHITECTURE_FIELDS:
            if key in saved and key in target:
                a, b = saved[key], target[key]
                if key == 'img_size':
                    a, b = tuple(a), tuple(b)
                if a != b:
                    raise RuntimeError(f'Checkpoint configuration mismatch for {key}: {a} != {b}')

    def load(self, f=None, warm_start=False):
        if not f:
            raise ValueError('A checkpoint path is required')
        checkpoint = self._load_file(f)
        trained_scfp = (checkpoint_mode(checkpoint) != 'baseline'
                        or any(k.removeprefix('module.').startswith(NEW_PREFIXES) for k in checkpoint['model']))
        if not warm_start or trained_scfp:
            self._validate_metadata(checkpoint)
        self.last_report = load_state_dict(self.model, checkpoint['model'], allow_new=warm_start and not trained_scfp)
        return {k: v for k, v in checkpoint.items() if k not in ('model', 'optimizer', 'scheduler', 'rng_state')}

    def resume(self, f=None):
        checkpoint = self._load_file(f)
        self._validate_metadata(checkpoint)
        model_args = vars(unwrap(self.model).args)
        saved = checkpoint.get('config', {})
        # Losses, protocol, split and schedule must not change silently on resume.
        for key in ('training_protocol', 'seed', 'validation_fraction', 'val_dataset',
                    'num_epoch', 'warmup_epochs', 'lrscheduler', 'scfp_slot_loss_weight',
                    'scfp_diversity_weight', 'scfp_slot_dropout', 'freeze_backbone', 'precision',
                    'batch_size', 'sampler', 'num_instance', 'img_aug', 'max_steps',
                    'optimizer', 'lr', 'scfp_lr', 'lr_factor', 'bias_lr_factor',
                    'weight_decay', 'weight_decay_bias', 'alpha', 'beta', 'momentum',
                    'milestones', 'gamma', 'warmup_factor', 'warmup_method',
                    'target_lr', 'power', 'temperature', 'id_loss_weight', 'mlm_loss_weight'):
            a, b = saved.get(key), model_args.get(key)
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                a, b = list(a), list(b)
            if key in saved and a != b:
                raise RuntimeError(f'Resume configuration mismatch: {key}')
        self.last_report = load_state_dict(self.model, checkpoint['model'])
        for key, component in (('optimizer', self.optimizer), ('scheduler', self.scheduler)):
            if component is not None:
                if key not in checkpoint:
                    raise RuntimeError(f'Resume requires {key} state')
                component.load_state_dict(checkpoint[key])
        if 'epoch' not in checkpoint:
            raise RuntimeError('Resume requires a saved epoch')
        if 'rng_state' in checkpoint:
            restore_rng(checkpoint['rng_state'])
        return {k: v for k, v in checkpoint.items() if k not in ('model', 'optimizer', 'scheduler', 'rng_state')}
