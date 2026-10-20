import pytest
import torch

from model.lgsp import FrequencySpatialPrompt, LocalGlobalSpatialFusion


@pytest.mark.parametrize('shape', [(2, 3, 384, 128), (2, 3, 224, 224), (1, 3, 31, 19), (1, 3, 1, 1)])
def test_frequency_contract_gradients(shape):
    torch.manual_seed(7)
    module = FrequencySpatialPrompt()
    x = torch.randn(shape, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape and y.device == x.device and y.dtype == x.dtype
    assert y.isfinite().all()
    (y * torch.randn_like(y)).sum().backward()
    for p in module.parameters():
        assert p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
    torch.testing.assert_close(module.train()(x), module.eval()(x), rtol=0, atol=0)
    assert module.complex_residual(x).imag.abs().max() < 1e-6


def test_partition_symmetry_cache_and_disable():
    module = FrequencySpatialPrompt()
    masks = module.ring_masks(31, 20, 'cpu')
    assert masks.data_ptr() == module.ring_masks(31, 20, 'cpu').data_ptr()
    torch.testing.assert_close(masks.sum(0), torch.ones(31, 20))
    x = torch.randn(2, 3, 31, 20)
    module.alpha.data.zero_()
    assert torch.equal(module(x), torch.zeros_like(x))
    assert '_rings' not in module.state_dict()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_fp32_fft_under_autocast(dtype):
    fusion = LocalGlobalSpatialFusion('frequency')
    x = torch.randn(1, 3, 33, 17).to(dtype)
    assert fusion.frequency(x).dtype == dtype
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = fusion(x)
    assert y.dtype == dtype and y.isfinite().all()
    y.float().square().mean().backward()
    assert fusion.frequency.logits.grad.isfinite().all()


def test_original_variant_negative_input_and_formula():
    x = -torch.ones(1, 3, 16, 8)
    m = FrequencySpatialPrompt(variant='lgsp_original', alpha_init=0.1)
    z = m.complex_residual(x)
    torch.testing.assert_close(m(x), 0.1 * (z.abs() - x))
    assert m(x).mean() > 0.1
    assert FrequencySpatialPrompt()(x).mean() < 0


def test_baseline_is_exact_identity():
    x = torch.randn(2, 3, 16, 8)
    m = LocalGlobalSpatialFusion()
    assert m(x) is x and not m.state_dict()
