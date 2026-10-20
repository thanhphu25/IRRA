import pytest
import torch

from model.scfp.feature_frequency import AdaptiveFeatureFrequency, radial_masks


@pytest.mark.parametrize('grid,width', [((24, 8), 768), ((9, 5), 32)])
def test_shape_symmetry_adaptivity_gradients(grid, width):
    torch.manual_seed(4)
    m = AdaptiveFeatureFrequency(width)
    m.capture = True
    x = torch.randn(2, 1 + grid[0] * grid[1], width, requires_grad=True)
    y = m(x, grid)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert torch.equal(x[:, 0], y[:, 0])
    assert m.last_diagnostics['frequency_imaginary_max'] < 1e-5
    assert m.last_diagnostics['frequency_residual_ratio'] < 0.01
    logits = m.last_features['frequency_logits']
    logits.retain_grad()
    assert (m.last_features['frequency_weights'][0] - m.last_features['frequency_weights'][1]).abs().max() > 1e-7
    y.square().mean().backward()
    assert logits.grad.abs().sum() > 0
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    masks = radial_masks(*grid, 4)
    assert torch.equal(masks.sum(0), torch.ones(grid))


@pytest.mark.parametrize('alpha', [0.0, 1e-8, 0.01])
def test_fft_identity_dtype_and_safe_scales(alpha):
    m = AdaptiveFeatureFrequency(16, alpha_init=alpha)
    x = torch.randn(2, 36, 16).half().requires_grad_()
    y = m(x, (7, 5))
    assert y.dtype == torch.float16 and torch.isfinite(y).all()
    torch.testing.assert_close(y, x, atol=0.002, rtol=0.002)
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    if alpha == 0:
        assert m.alpha_raw.grad.abs() > 0  # gate gradients resume after alpha moves


def test_fft_reconstruction_extreme_gains_and_bad_shapes():
    grid = torch.randn(2, 8, 9, 5)
    spectrum = torch.fft.fft2(grid, norm='ortho')
    restored = torch.fft.ifft2(spectrum, norm='ortho')
    torch.testing.assert_close(restored.real, grid)
    assert restored.imag.abs().max() < 1e-6
    m = AdaptiveFeatureFrequency(8)
    with torch.no_grad():
        m.alpha_raw.fill_(100)
        m.gate.net[-1].bias.copy_(torch.tensor([100., -100, -100, -100]))
    out = m(grid.flatten(2).transpose(1, 2).new_zeros(2, 46, 8), (9, 5))
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError):
        m(torch.randn(2, 10, 8), (9, 5))


def test_fft_is_float32_under_autocast(monkeypatch):
    calls = []
    original = torch.fft.fft2
    def checked(x, *args, **kwargs):
        calls.append(x.dtype)
        return original(x, *args, **kwargs)
    monkeypatch.setattr(torch.fft, 'fft2', checked)
    module = AdaptiveFeatureFrequency(16)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = module(torch.randn(2, 46, 16).half(), (9, 5))
    assert calls == [torch.float32] and result.dtype == torch.float16
