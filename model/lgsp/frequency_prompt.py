"""LGSP-inspired Fourier residuals. See THIRD_PARTY_NOTICES.md for attribution."""
import torch
from torch import nn


class FrequencySpatialPrompt(nn.Module):
    """Return a *scaled residual*, never an enhanced image.

    Operates on IRRA's CLIP-normalized RGB, without denormalization or clamping.
    real: alpha * real(IFFT(FFT(x) * sum(softmax(logits/T) * rings))).
    lgsp_original: alpha * (abs(IFFT(FFT(x) * (1 + mask))) - x).
    The latter preserves the published code's abs transformation, but adapts
    its radii to H,W and uses a small learnable alpha; it is not full LGSP.
    """
    def __init__(self, num_rings=8, temperature=1.0, alpha_init=1e-3, variant="real"):
        super().__init__()
        if num_rings < 2 or temperature <= 0:
            raise ValueError("num_rings >= 2 and temperature > 0 required")
        if variant not in ("real", "lgsp_original"):
            raise ValueError(f"Unknown frequency variant: {variant}")
        self.num_rings, self.temperature, self.variant = num_rings, temperature, variant
        self.logits = nn.Parameter(torch.zeros(num_rings))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.register_buffer("_rings", torch.empty(0), persistent=False)
        self._mask_key = None

    def ring_weights(self):
        # Original LGSP multiplies by temperature, conventional variant divides.
        scores = (self.logits.float() * self.temperature if self.variant == "lgsp_original"
                  else self.logits.float() / self.temperature)
        return scores.softmax(0)

    def ring_masks(self, height, width, device):
        key = (height, width, torch.device(device))
        if key != self._mask_key or self._rings.dtype != torch.float32:
            y = torch.arange(height, device=device, dtype=torch.float32) - height // 2
            x = torch.arange(width, device=device, dtype=torch.float32) - width // 2
            distance = (y[:, None].square() + x[None, :].square()).sqrt()
            radius = max(((height // 2)**2 + (width // 2)**2)**0.5, 1.0)
            if self.variant == "lgsp_original":
                outer = torch.linspace(0, radius, self.num_rings, device=device)
                inner = torch.cat((outer[:1], outer[:-1] + 1e-6))
                masks = torch.sigmoid(4 * (outer[:, None, None] - distance)) - \
                    torch.sigmoid(4 * (inner[:, None, None] - distance))
            else:
                # Smooth partition of unity, including DC and corner frequencies.
                edges = torch.linspace(0, radius, self.num_rings + 1, device=device)[1:-1]
                disks = torch.sigmoid(4 * (edges[:, None, None] - distance))
                masks = torch.cat((disks[:1], disks[1:] - disks[:-1], 1 - disks[-1:]))
            self._rings = masks
            self._mask_key = key
        return self._rings

    def frequency_mask(self, height, width, device):
        return (self.ring_weights()[:, None, None] * self.ring_masks(height, width, device)).sum(0)

    def complex_residual(self, x):
        """Expose complex IFFT for symmetry diagnostics; always complex64."""
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("Expected floating RGB tensor [B,3,H,W]")
        if not x.is_floating_point():
            raise ValueError("Expected floating RGB tensor [B,3,H,W]")
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
            mask = self.frequency_mask(*x.shape[-2:], x.device)
            if self.variant == "lgsp_original":
                mask = 1 + mask
            return torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)))

    def residual(self, x):
        z = self.complex_residual(x)
        return z.abs() - x.float() if self.variant == "lgsp_original" else z.real

    def forward(self, x):
        return (self.alpha.float() * self.residual(x)).to(x.dtype)
