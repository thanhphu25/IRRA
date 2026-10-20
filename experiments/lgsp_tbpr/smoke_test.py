"""One real-dataset batch through SDM/MLM/ID, optimizer, and prompt diagnostics.

Run from repo root: python -m experiments.lgsp_tbpr.smoke_test [train.py args]
"""
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from datasets import build_dataloader
from model import build_model
from solver import build_optimizer
from train import set_seed
from utils.checkpoint import Checkpointer
from utils.options import get_args


def run_smoke(args):
    args.distributed = False
    args.lgsp_diagnostics = True
    set_seed(args.seed, args.deterministic)
    loader, _, _, num_classes = build_dataloader(args)
    model = build_model(args, num_classes).to(args.device)
    if args.init_checkpoint:
        Checkpointer(model).warm_start(args.init_checkpoint)
    batch = {k: v.to(args.device) for k, v in next(iter(loader)).items()}
    cuda = next(model.parameters()).device.type == 'cuda'
    def sync():
        if cuda:
            torch.cuda.synchronize()

    # Same weights, toggled bypass; no second backbone allocation.
    model.eval()
    mode = model.spatial_prompt.mode
    with torch.no_grad():
        enhanced = model.encode_image(batch['images'])
        text = model.encode_text(batch['caption_ids'])
        model.spatial_prompt.mode = 'baseline'
        original = model.encode_image(batch['images'])
        model.spatial_prompt.mode = mode
        image_feature_shape = list(enhanced.shape)
        image_difference = (model.transform_image(batch['images']) - batch['images']).abs().max().item()
    initial = dict(cosine_similarity=F.cosine_similarity(original, enhanced).mean().item(),
                   max_embedding_difference=(enhanced - original).abs().max().item(),
                   max_image_difference=image_difference)
    del original, enhanced
    optimizer = build_optimizer(args, model)
    model.train()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    sync(); start = time.perf_counter()
    ret = model(batch)
    loss = sum(v for k, v in ret.items() if 'loss' in k)
    sync(); forward_seconds = time.perf_counter() - start
    if not loss.isfinite():
        raise FloatingPointError(f'Nonfinite smoke loss: {ret}')
    optimizer.zero_grad()
    start = time.perf_counter()
    loss.backward()
    sync(); backward_seconds = time.perf_counter() - start
    gradients = {}
    for name, p in model.spatial_prompt.named_parameters():
        if p.grad is None or not p.grad.isfinite().all():
            raise RuntimeError(f'Missing/nonfinite prompt gradient: {name}')
        gradients[name] = p.grad.norm().item()
    optimizer.step()
    result = dict(mode=mode, backbone=args.pretrain_choice, input_shape=list(batch['images'].shape),
                  image_feature_shape=image_feature_shape, text_feature_shape=list(text.shape),
                  losses={k: v.detach().item() for k, v in ret.items() if 'loss' in k},
                  total_loss=loss.detach().item(), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  prompt_gradients=gradients, peak_allocated_vram_gb=torch.cuda.max_memory_allocated() / 1024**3 if cuda else None,
                  forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                  near_identity=initial, diagnostics=model.spatial_prompt.last_diagnostics,
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                      python=platform.python_version(), device=torch.cuda.get_device_name() if cuda else 'cpu'))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'smoke_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    run_smoke(get_args())
