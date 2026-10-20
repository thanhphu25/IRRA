import torch
import torch.nn.functional as F


def slot_diversity(slots):
    k = slots.shape[1]
    if k == 1:
        return slots.sum() * 0
    slots = F.normalize(slots.float(), dim=-1)
    similarity = slots @ slots.transpose(-1, -2)
    off_diagonal = ~torch.eye(k, dtype=torch.bool, device=slots.device)
    return similarity[:, off_diagonal].square().mean()
