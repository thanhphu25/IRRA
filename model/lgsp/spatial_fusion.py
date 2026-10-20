import torch
from torch import nn

from .frequency_prompt import FrequencySpatialPrompt
from .local_prompt import LocalSpatialPrompt


class LocalGlobalSpatialFusion(nn.Module):
    """Branches return scaled residuals; add original x exactly once."""
    def __init__(self, mode="baseline", diagnostics=False, **kwargs):
        super().__init__()
        if mode not in ("baseline", "frequency", "local", "both"):
            raise ValueError(f"Unsupported LGSP mode: {mode}")
        self.mode = mode
        self.frequency = self.local = None
        self.diagnostics_enabled = diagnostics
        self.last_diagnostics = {}
        if mode in ("frequency", "both"):
            self.frequency = FrequencySpatialPrompt(
                kwargs.get("frequency_num_rings", 8), kwargs.get("frequency_temperature", 1.0),
                kwargs.get("frequency_alpha_init", 1e-3), kwargs.get("frequency_variant", "real"))
        if mode in ('local', 'both'):
            self.local = LocalSpatialPrompt(
                pool_size=kwargs.get('local_pool_size', 4), hidden_dim=kwargs.get('local_hidden_dim', 8),
                kernel_size=kwargs.get('local_kernel_size', 3), selection=kwargs.get('local_selection', 'softmax'),
                top_k=kwargs.get('local_top_k', 2), temperature=kwargs.get('local_temperature', 1.0),
                dropout=kwargs.get('local_dropout', 0.0), alpha_init=kwargs.get('local_alpha_init', 1e-3),
                checkpoint_candidates=kwargs.get('local_checkpoint', False))

    def forward(self, x):
        if self.mode == "baseline":
            return x
        enhanced = x.float()
        stats = {}
        denominator = x.detach().float().norm().clamp_min(1e-12) if self.diagnostics_enabled else None
        for name in ('local', 'frequency'):
            branch = getattr(self, name)
            if branch is None:
                continue
            if name == 'local' and self.diagnostics_enabled:
                contribution, weights = branch(x, return_weights=True)
                stats['local_weights'] = weights.detach().mean(dim=(0, 2, 3)).cpu().tolist()
                selected = weights.detach().argmax(1)
                stats['local_utilization'] = torch.bincount(selected.flatten(), minlength=branch.pool_size).float().div(selected.numel()).cpu().tolist()
            else:
                contribution = branch(x)
            enhanced = enhanced + contribution
            if self.diagnostics_enabled:
                stats[name + '_alpha'] = branch.alpha.detach().item()
                norm = contribution.detach().float().norm()
                stats[name + '_contribution_norm'] = norm.item()
                stats[name + '_contribution_ratio'] = (norm / denominator).item()
                stats[name + '_residual_norm'] = (norm / branch.alpha.detach().abs().clamp_min(1e-12)).item()
        if self.diagnostics_enabled:
            if self.frequency is not None:
                stats['frequency_weights'] = self.frequency.ring_weights().detach().cpu().tolist()
            self.last_diagnostics = stats
        return enhanced.to(x.dtype)
