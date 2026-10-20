"""Lightweight shared-prototype cross attention; not PLOT's iterative PSA/GRU."""
import torch
from torch import nn
import torch.nn.functional as F


def text_content_mask(ids, sot_id=49406, eot_id=49407, mask_id=49405):
    eot = ids.eq(eot_id)
    if not eot.any(dim=-1).all():
        raise ValueError('Each tokenized description must contain EOT')
    end = eot.int().argmax(dim=-1)
    position = torch.arange(ids.shape[1], device=ids.device)[None]
    return ((position > 0) & (position < end[:, None]) & ids.ne(0)
            & ids.ne(sot_id) & ids.ne(eot_id) & ids.ne(mask_id))


class SemanticSlots(nn.Module):
    def __init__(self, dim=512, heads=4, dropout=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.query_projection.weight)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, tokens, shared_queries, valid=None):
        tokens = tokens.float()
        q = self.query_projection(self.query_norm(shared_queries))
        q = q.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        if valid is None:
            valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        # A zero sentinel for empty descriptions prevents all-masked softmax NaNs.
        # No padding or special-token feature is ever used as semantic content.
        empty = ~valid.any(-1)
        safe_valid = valid.clone()
        safe_valid[empty, 0] = True
        values = self.token_norm(tokens).masked_fill(~valid[:, :, None], 0)
        x, attention = self.attention(q, values, values, key_padding_mask=~safe_valid,
                                      need_weights=True, average_attn_weights=True)
        x = x + shared_queries.unsqueeze(0)
        x = x + self.ffn(x)
        slots = F.normalize(self.output_norm(x), dim=-1)
        attention = attention.masked_fill(~valid[:, None, :], 0)
        return slots, attention


class DualModalSemanticSlots(nn.Module):
    def __init__(self, dim=512, num_slots=4, heads=4, dropout=0.0, vocab_size=49408):
        super().__init__()
        if num_slots < 1 or heads < 1 or dim % heads or not 0 <= dropout < 1:
            raise ValueError('Invalid semantic slot dimensions/dropout')
        self.shared_queries = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.orthogonal_(self.shared_queries)
        self.visual = SemanticSlots(dim, heads, dropout)
        self.text = SemanticSlots(dim, heads, dropout)
        self.relevance_head = nn.Linear(dim, 1)
        self.vocab_size = vocab_size

    def forward(self, image_tokens, text_tokens, ids):
        with torch.autocast(device_type=image_tokens.device.type, enabled=False):
            mask = text_content_mask(ids, self.vocab_size - 2, self.vocab_size - 1,
                                     self.vocab_size - 3)
            visual, va = self.visual(image_tokens.float(), self.shared_queries)
            text, ta = self.text(text_tokens.float(), self.shared_queries, mask)
            # Positive floor + sum normalization preclude the all-zero solution.
            raw = 0.1 + 0.9 * self.relevance_head(text).squeeze(-1).sigmoid()
            relevance = raw / raw.sum(-1, keepdim=True)
        return dict(visual_slots=visual, text_slots=text, slot_relevance=relevance,
                    slot_relevance_raw=raw, visual_attention=va, text_attention=ta,
                    text_content_mask=mask)
