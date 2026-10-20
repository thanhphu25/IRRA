"""Compact CNN prompt pool, inspired by LGSP's pixelwise cosine-weighted path."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LocalSpatialPrompt(nn.Module):
    """Return alpha times a weighted image-conditioned spatial residual.

    Selection is pixelwise across the pool. There is no extra CLIP pass, text
    prompt, task key, or anatomical supervision. Candidates are kept as a list
    to avoid an additional [B,pool,3,H,W] allocation during weighted reduction.
    """
    def __init__(self, pool_size=4, hidden_dim=8, kernel_size=3, selection='softmax',
                 top_k=2, temperature=1.0, dropout=0.0, alpha_init=1e-3,
                 checkpoint_candidates=False):
        super().__init__()
        if pool_size < 1 or hidden_dim < 1 or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('Positive pool/hidden sizes and a positive odd kernel required')
        if selection not in ('softmax', 'topk') or (selection == 'topk' and not 1 <= top_k <= pool_size):
            raise ValueError('selection must be softmax/topk and 1 <= top_k <= pool_size')
        if temperature <= 0 or not 0 <= dropout < 1:
            raise ValueError('temperature > 0 and 0 <= dropout < 1 required')
        self.pool_size, self.selection, self.top_k = pool_size, selection, top_k
        self.temperature, self.checkpoint_candidates = temperature, checkpoint_candidates
        self.generators = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, hidden_dim, kernel_size, padding=kernel_size // 2),
                          nn.ReLU(),
                          nn.Conv2d(hidden_dim, 3, kernel_size, padding=kernel_size // 2),
                          nn.Dropout(dropout)) for _ in range(pool_size)])
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def candidates(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or not x.is_floating_point():
            raise ValueError('Expected floating RGB tensor [B,3,H,W]')
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            return [checkpoint(g, x, use_reentrant=False) if
                    self.checkpoint_candidates and self.training and torch.is_grad_enabled()
                    else g(x) for g in self.generators]

    def selection_weights(self, x, candidates):
        scores = torch.stack([F.cosine_similarity(x.float(), p, dim=1, eps=1e-6)
                              for p in candidates], dim=1) / self.temperature
        if self.selection == 'topk':
            indices = scores.topk(self.top_k, dim=1).indices
            selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, indices, True)
            scores = scores.masked_fill(~selected, float('-inf'))
        return scores.softmax(1)

    def forward(self, x, return_weights=False):
        with torch.autocast(device_type=x.device.type, enabled=False):
            candidates = self.candidates(x)
            weights = self.selection_weights(x, candidates)
            residual = sum(p * weights[:, i:i + 1] for i, p in enumerate(candidates))
            result = (self.alpha.float() * residual).to(x.dtype)
        return (result, weights) if return_weights else result
