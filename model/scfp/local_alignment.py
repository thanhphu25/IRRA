"""Identity-aware bidirectional local SDM, matching IRRA's distribution objective."""
import torch
import torch.nn.functional as F


def positive_targets(pids):
    positive = pids.reshape(-1, 1).eq(pids.reshape(1, -1)).float()
    return positive / positive.sum(-1, keepdim=True)


def local_similarity(visual, text, relevance):
    if visual.shape != text.shape or relevance.shape != text.shape[:2]:
        raise ValueError('Expected matching [B,K,D] slots and [B,K] relevance')
    visual, text = F.normalize(visual.float(), dim=-1), F.normalize(text.float(), dim=-1)
    relevance = relevance.float().clamp_min(1e-8)
    relevance = relevance / relevance.sum(-1, keepdim=True)
    return torch.einsum('ikd,jkd,jk->ij', visual, text, relevance)


def local_alignment_loss(similarity, pids, logit_scale):
    if similarity.shape != (pids.numel(), pids.numel()) or pids.numel() == 0:
        raise ValueError('Local scores must be [B,B] for a nonempty identity batch')
    targets = positive_targets(pids)
    logits = similarity.float() * logit_scale
    def direction(scores):
        log_pred = F.log_softmax(scores, dim=-1)
        return (log_pred.exp() * (log_pred - (targets + 1e-8).log())).sum(-1).mean()
    return direction(logits) + direction(logits.t())
