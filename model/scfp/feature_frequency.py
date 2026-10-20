"""Conjugate-symmetric, bounded, image-adaptive FFT on intermediate ViT patches."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .frequency_gate import FrequencyGate


def radial_masks(height, width, bands, device=None):
    """Disjoint hard bands in normalized cycles/pixel; no fftshift is needed.

    Radius is sqrt((fy/.5)^2 + (fx/.5)^2)/sqrt(2). This handles rectangular
    grids and assigns every bin exactly once, including DC/Nyquist/corners.
    Absolute fftfreq coordinates preserve the Hermitian partner of every bin.
    """
    if min(height, width, bands) < 1:
        raise ValueError("Positive grid dimensions and band count required")
    fy = torch.fft.fftfreq(height, device=device).abs() / 0.5
    fx = torch.fft.fftfreq(width, device=device).abs() / 0.5
    radius = (fy[:, None].square() + fx[None, :].square()).sqrt() / math.sqrt(2)
    indices = (radius * bands).long().clamp_max(bands - 1)
    return F.one_hot(indices, bands).permute(2, 0, 1).float()


class AdaptiveFeatureFrequency(nn.Module):
    def __init__(self, width, bands=4, hidden_dim=64, temperature=1.0,
                 alpha_init=0.01):
        super().__init__()
        if not 0 <= abs(alpha_init) < 1:
            raise ValueError("abs(alpha_init) must be < 1")
        self.bands = bands
        self.gate = FrequencyGate(width, bands, hidden_dim, temperature)
        # G = 1 + tanh(alpha_raw) * tanh(R*w - 1), hence 0 < G < 2.
        self.alpha_raw = nn.Parameter(torch.tensor(math.atanh(alpha_init)))
        self.enabled = True
        self.capture = False
        self.last_features = {}
        self.last_diagnostics = {}

    @property
    def alpha(self):
        return self.alpha_raw.tanh()

    def forward(self, tokens, grid_size):
        """tokens: [B,1+H*W,C]. CLS is copied untouched at this operation."""
        if not self.enabled:
            return tokens
        h, w = grid_size
        if tokens.ndim != 3 or tokens.shape[1] != 1 + h * w:
            raise ValueError("Token count must equal 1 + grid height * width")
        dtype = tokens.dtype
        # Explicitly disable autocast: FFT, gate and residual arithmetic are FP32.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            grid = tokens[:, 1:].float().transpose(1, 2).reshape(tokens.shape[0], -1, h, w)
            logits, weights = self.gate(grid)
            masks = radial_masks(h, w, self.bands, grid.device)
            response = 1 + self.alpha * torch.einsum(
                'br,rhw->bhw', (self.bands * weights - 1).tanh(), masks)
            spectrum = torch.fft.fft2(grid, dim=(-2, -1), norm='ortho')
            restored = torch.fft.ifft2(spectrum * response[:, None], dim=(-2, -1), norm='ortho')
            enhanced = restored.real
            residual = enhanced - grid
            # Only detached small statistics persist during ordinary training.
            with torch.no_grad():
                self.last_diagnostics = {
                    'frequency_alpha': self.alpha.detach(),
                    'frequency_weights_mean': weights.mean(0).detach(),
                    'frequency_weights_std': weights.std(0, unbiased=False).detach(),
                    'frequency_weights_entropy': (-(weights * weights.clamp_min(1e-8).log()).sum(-1).mean()).detach(),
                    'frequency_gate_variance_across_images': weights.var(0, unbiased=False).mean().detach(),
                    'frequency_residual_norm': residual.norm().detach(),
                    'frequency_residual_ratio': (residual.norm() / grid.norm().clamp_min(1e-8)).detach(),
                    'frequency_imaginary_max': restored.imag.abs().amax().detach(),
                }
            self.last_features = {}
            if self.capture:
                # Optional differentiable interface. Compute bands one at a time
                # to avoid a [B,R,C,H,W] complex allocation in normal training.
                band_features = torch.stack([
                    torch.fft.ifft2(spectrum * m, dim=(-2, -1), norm='ortho').real
                    .square().flatten(2).mean(-1).clamp_min(1e-12).sqrt() for m in masks], dim=1)
                self.last_features = dict(frequency_logits=logits, frequency_weights=weights,
                                          frequency_band_features=band_features,
                                          frequency_feature_shape=list(grid.shape))
            patches = enhanced.flatten(2).transpose(1, 2).to(dtype)
            return torch.cat((tokens[:, :1], patches), dim=1)
