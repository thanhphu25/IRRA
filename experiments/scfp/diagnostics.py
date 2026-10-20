"""Detached diagnostics and reproducible run provenance; no research claims."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch


def serializable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + '\n')


def append_jsonl(path, value):
    with open(path, 'a') as stream:
        stream.write(json.dumps(serializable(value), allow_nan=False) + '\n')


def gradient_norms(model):
    groups = dict(visual_slots=[], text_slots=[], shared_queries=[], frequency_gate=[],
                  frequency_residual_scale=[], vision_backbone=[], text_backbone=[])
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if 'scfp_frequency.gate.' in name:
            key = 'frequency_gate'
        elif 'scfp_frequency.alpha_raw' in name:
            key = 'frequency_residual_scale'
        elif name.startswith('scfp_slots.visual.'):
            key = 'visual_slots'
        elif name.startswith('scfp_slots.text.'):
            key = 'text_slots'
        elif name.startswith('scfp_slots.'):
            key = 'shared_queries'
        elif name.startswith('base_model.visual.'):
            key = 'vision_backbone'
        elif name.startswith('base_model.'):
            key = 'text_backbone'
        else:
            continue
        groups[key].append(p.grad.detach().float().square().sum())
    norms = {key: torch.stack(values).sum().sqrt() if values else 0.0 for key, values in groups.items()}
    norms['frequency_gradient_norm'] = norms['frequency_gate']
    return serializable(norms)


def module_diagnostics(model, pids=None):
    result = {}
    frequency = getattr(model.base_model.visual, 'scfp_frequency', None)
    if frequency is not None:
        result.update(frequency.last_diagnostics)
    features = getattr(model, 'scfp_features', {})
    for modality in ('visual', 'text'):
        slots = features.get(f'{modality}_slots')
        attention = features.get(f'{modality}_attention')
        if slots is not None:
            result[f'{modality}_slot_similarity'] = (slots @ slots.transpose(-1, -2)).mean(0)
        if attention is not None:
            result[f'{modality}_attention_entropy'] = -(attention * attention.clamp_min(1e-8).log()).sum(-1).mean()
    if 'slot_relevance' in features:
        result['slot_relevance'] = features['slot_relevance'].mean(0)
        result['slot_relevance_raw'] = features['slot_relevance_raw'].mean(0)
    if pids is not None and 'local_similarity' in features:
        similarity = features['local_similarity']
        mask = pids[:, None].eq(pids[None, :])
        pos = similarity[mask].mean()
        result['positive_local_similarity'] = pos
        if (~mask).any():
            neg = similarity[~mask].mean()
            result['negative_local_similarity'] = neg
            result['local_similarity_margin'] = pos - neg
        visual, text = features['visual_slots'], features['text_slots']
        matching = torch.einsum('bkd,bld->bkl', visual, text).mean(0)
        result['paired_slot_matching_matrix'] = matching
        result['paired_same_index_argmax_fraction'] = matching.argmax(-1).eq(torch.arange(matching.shape[0], device=matching.device)).float().mean()
    return serializable(result)


@torch.no_grad()
def fixed_sample_diagnostics(model, batch, output_dir, epoch):
    was_training = model.training
    model.eval()
    _, features = model(batch, return_features=True)
    result = module_diagnostics(model, batch['pids'])
    frequency = getattr(model.base_model.visual, 'scfp_frequency', None)
    if frequency is not None:
        result['fixed_sample_frequency_weights'] = serializable(features['frequency_weights'])
        result['fixed_sample_ids'] = serializable(batch.get('image_ids', torch.arange(len(batch['images']))))
    result['epoch'] = epoch
    write_json(Path(output_dir) / 'diagnostics.json', result)
    # Save optional slot maps / band energy without bloating JSON logs.
    torch.save({k: v.detach().cpu() for k, v in features.items() if torch.is_tensor(v)},
               Path(output_dir) / 'fixed_sample_features.pt')
    if frequency is not None:
        frequency.capture = False
        frequency.last_features = {}
    model.train(was_training)
    return result


def environment(args):
    def git(*command):
        result = subprocess.run(['git', *command], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    root = Path(__file__).resolve().parents[2]
    source_hash = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        if '.git' not in path.parts and '.venv' not in path.parts:
            source_hash.update(str(path.relative_to(root)).encode())
            source_hash.update(path.read_bytes())
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'PyYAML'):
        packages[name] = importlib.metadata.version(name)
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                git_commit=git('rev-parse', 'HEAD'), git_branch=git('branch', '--show-current'),
                git_status=git('status', '--short'), source_sha256=source_hash.hexdigest(),
                protocol=args.training_protocol, init_checkpoint=args.init_checkpoint,
                resume_checkpoint=args.resume_checkpoint)
