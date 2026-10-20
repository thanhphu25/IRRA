import pytest
import torch
from model.lgsp import LocalGlobalSpatialFusion


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device unavailable')
@pytest.mark.parametrize('mode', ['local', 'frequency', 'both'])
def test_cuda_fp16_fft_gradients(mode):
    m = LocalGlobalSpatialFusion(mode).cuda()
    x = torch.randn(2, 3, 384, 128, device='cuda', dtype=torch.float16)
    with torch.autocast('cuda', dtype=torch.float16):
        y = m(x)
    assert y.shape == x.shape and y.dtype == x.dtype and y.isfinite().all()
    y.float().square().mean().backward()
    for p in m.parameters():
        assert p.grad is not None and p.grad.isfinite().all()
