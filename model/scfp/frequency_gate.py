import torch
from torch import nn


class FrequencyGate(nn.Module):
    def __init__(self, width, bands=4, hidden_dim=64, temperature=1.0):
        super().__init__()
        if min(width, bands, hidden_dim) < 1 or temperature <= 0:
            raise ValueError("Positive gate dimensions and temperature required")
        self.temperature = temperature
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, bands))
        # Nonzero output weights allow gradients through BOTH MLP layers at step 1.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, grid):
        logits = self.net(grid.mean(dim=(-2, -1)))
        return logits, torch.softmax(logits / self.temperature, dim=-1)
