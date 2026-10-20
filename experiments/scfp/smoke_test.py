"""One-batch V2 forward/backward + checkpoint and near-identity diagnostics.

Use --tiny for an offline CPU engineering check. The actual Kaggle smoke test
uses --config_file ORIGINAL/configs.yaml --init_checkpoint ORIGINAL/best.pth.
"""
import argparse
import importlib
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model.clip_model import CLIP
from utils.options import get_args
from utils.checkpoint import Checkpointer
from solver import build_optimizer
from experiments.scfp.diagnostics import gradient_norms, module_diagnostics, write_json


def tiny_clip(*args, **kwargs):
    config = dict(embed_dim=512, image_resolution=(32, 16), vision_layers=3,
                  vision_width=64, vision_patch_size=8, stride_size=8,
                  context_length=8, vocab_size=128, transformer_width=64,
                  transformer_heads=1, transformer_layers=1)
    return CLIP(**config), config


def synthetic_batch(args, device, size=4, num_classes=3):
    ids = torch.zeros(size, args.text_length, dtype=torch.long, device=device)
    ids[:, 0] = args.vocab_size - 2
    ids[:, 1:3] = torch.randint(1, min(100, args.vocab_size - 3), (size, 2), device=device)
    ids[:, 3] = args.vocab_size - 1
    masked, labels = ids.clone(), torch.zeros_like(ids)
    masked[:, 1], labels[:, 1] = args.vocab_size - 3, ids[:, 1]
    return dict(images=torch.randn(size, 3, *args.img_size, device=device), caption_ids=ids,
                mlm_ids=masked, mlm_labels=labels,
                pids=(torch.arange(size, device=device) // 2) % num_classes)


def run(args, output, tiny=False, synthetic=False, config_file=None):
    torch.manual_seed(args.seed)
    if args.device == 'cpu':
        torch.set_num_threads(2)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output.parent)
    args.distributed = False
    # Reuse ORIGINAL architecture only, never old LGSP configuration behavior.
    if config_file:
        from utils.iotools import load_train_configs
        original = load_train_configs(config_file)
        for key in ('pretrain_choice', 'img_size', 'stride_size', 'text_length', 'vocab_size',
                    'cmt_depth', 'loss_names', 'id_loss_weight', 'mlm_loss_weight', 'temperature'):
            if key in original:
                setattr(args, key, original[key])
    if tiny:
        args.img_size, args.text_length, args.vocab_size = (32, 16), 8, 128
        args.cmt_depth, args.scfp_frequency_layer, args.precision = 1, 1, 'fp32'
        num_classes = 3
    elif synthetic:
        if not args.init_checkpoint:
            raise ValueError('Full synthetic smoke requires init_checkpoint or use --tiny')
        state = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        state = state.get('model', state)
        key = next(k for k in state if k.removeprefix('module.') == 'classifier.weight')
        num_classes = state[key].shape[0]
    else:
        from datasets import build_dataloader
        # A smoke batch does not use any validation/test queries.
        args.val_dataset = 'none'
        loader, _, _, num_classes = build_dataloader(args)
    build = importlib.import_module('model.build')
    with patch.object(build, 'build_CLIP_from_openai_pretrained', tiny_clip) if tiny else _nullcontext():
        model = build.build_model(args, num_classes).to(args.device)
    cp = Checkpointer(model)
    if args.init_checkpoint:
        cp.load(args.init_checkpoint, warm_start=True)
    optimizer = build_optimizer(args, model)
    batch = (synthetic_batch(args, args.device, size=4 if tiny else args.batch_size, num_classes=num_classes)
             if tiny or synthetic else {k: v.to(args.device) for k, v in next(iter(loader)).items()})
    cuda = args.device.startswith('cuda')
    def sync():
        if cuda:
            torch.cuda.synchronize()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    sync()
    started = time.perf_counter()
    ret, features = model(batch, return_features=True)
    sync()
    forward_time = time.perf_counter() - started
    total = sum(v for k, v in ret.items() if k.endswith('_loss'))
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite smoke loss')
    started = time.perf_counter()
    total.backward()
    sync()
    backward_time = time.perf_counter() - started
    gradients = gradient_norms(model)
    for name, p in model.named_parameters():
        if p.requires_grad and 'scfp_' in name:
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError(f'Missing/nonfinite gradient: {name}')
    diagnostics = module_diagnostics(model, batch['pids'])
    model.eval()
    frequency = getattr(model.base_model.visual, 'scfp_frequency', None)
    with torch.no_grad():
        enhanced = model.encode_image(batch['images'])
        if frequency is not None:
            frequency.enabled = False
            original = model.encode_image(batch['images'])
            frequency.enabled = True
        else:
            original = enhanced
        text = model.encode_text(batch['caption_ids'])
    report = dict(status='PASS', tiny=tiny, synthetic_data=tiny or synthetic,
                  original_checkpoint=args.init_checkpoint or None,
                  checkpoint_report=cp.last_report, mode=args.scfp_mode, backbone=args.pretrain_choice,
                  actual_backbone=dict(visual_width=model.base_model.visual.conv1.out_channels,
                                       visual_layers=len(model.base_model.visual.transformer.resblocks),
                                       embedding_dim=model.embed_dim),
                  image_shape=list(batch['images'].shape), text_shape=list(batch['caption_ids'].shape),
                  feature_shapes={k: list(v.shape) for k, v in features.items() if torch.is_tensor(v)},
                  losses={k: float(v.detach()) for k, v in ret.items() if k.endswith('_loss')},
                  total_loss=float(total.detach()), gradient_norms=gradients,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  peak_gpu_memory=torch.cuda.max_memory_allocated() if cuda else None,
                  forward_seconds=forward_time, backward_seconds=backward_time,
                  mean_embedding_cosine=F.cosine_similarity(original, enhanced).mean().item(),
                  max_embedding_difference=(original - enhanced).abs().max().item(), diagnostics=diagnostics)
    if frequency is not None:
        report['frequency_feature_shape'] = frequency.last_features.get('frequency_feature_shape')
    # Roundtrip after an actual step; initial embedding comparison is above.
    optimizer.step()
    with tempfile.TemporaryDirectory(prefix='scfp-smoke-') as directory:
        saved = Checkpointer(model, optimizer, save_dir=directory, save_to_disk=True)
        saved.save('smoke', epoch=1)
        saved.load(Path(directory) / 'smoke.pth')
    write_json(output, report)
    print(f"{args.scfp_mode} smoke PASS; loss={report['total_loss']:.6f}; report={output}")
    print(f"Initial embedding cosine={report['mean_embedding_cosine']:.9f}; max difference={report['max_embedding_difference']:.6g}")
    return report


def _nullcontext():
    from contextlib import nullcontext
    return nullcontext()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--synthetic_data', action='store_true')
    parser.add_argument('--config_file')
    parser.add_argument('--output_file', default='scfp_smoke.json')
    own, remaining = parser.parse_known_args(argv)
    if '--scfp_mode' not in remaining:
        remaining += ['--scfp_mode', 'v2']
    if '--training_protocol' not in remaining:
        warm = '--init_checkpoint' in remaining
        remaining += ['--training_protocol', 'warmstart' if warm else 'scientific',
                      '--val_dataset', 'none' if warm else 'holdout']
    args = get_args(remaining)
    return run(args, own.output_file, own.tiny, own.synthetic_data, own.config_file)


if __name__ == '__main__':
    main()
