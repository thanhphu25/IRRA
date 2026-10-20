import importlib

import pytest
import torch

from tests.helpers import tiny_args, tiny_clip, synthetic_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable locally; run on Kaggle T4')
@pytest.mark.parametrize('precision', ['fp16', 'fp32'])
def test_cuda_mixed_precision_forward_backward(monkeypatch, precision):
    build = importlib.import_module('model.build')
    monkeypatch.setattr(build, 'build_CLIP_from_openai_pretrained', tiny_clip)
    args = tiny_args()
    args.precision = precision
    model = build.build_model(args, 3).cuda()
    batch = {k: v.cuda() for k, v in synthetic_batch().items()}
    with torch.autocast('cuda', dtype=torch.float16):
        losses = model(batch)
        total = sum(v for k, v in losses.items() if k.endswith('_loss'))
    total.backward()
    assert torch.isfinite(total)
    assert model.base_model.visual.scfp_frequency.gate.net[-1].weight.dtype == torch.float32
    for name, p in model.named_parameters():
        if 'scfp_' in name:
            assert p.grad is not None and torch.isfinite(p.grad).all()
