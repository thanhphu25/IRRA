"""Exact retrieval metrics, query chunking migrated from the Kaggle notebook.

Metrics are percentages. Absent query IDs contribute zero, remaining in the
denominator. Ranks beyond gallery size repeat the last available CMC value.
"""
import logging
import warnings

import torch
import torch.nn.functional as F
from prettytable import PrettyTable


def _validate(q_pids, g_pids, max_rank):
    if max_rank < 1 or q_pids.numel() == 0 or g_pids.numel() == 0:
        raise ValueError("Retrieval needs nonempty query/gallery and max_rank >= 1")
    missing = int((~torch.isin(q_pids.cpu(), g_pids.cpu())).sum())
    if missing:
        warnings.warn(f'{missing} queries have no gallery positive; explicitly counted as zero', RuntimeWarning)


def _statistics(indices, q_pids, g_pids, max_rank, full):
    matches = g_pids.cpu().view(-1)[indices.cpu()].eq(q_pids.cpu().view(-1, 1))
    cmc = matches[:, :max_rank].cumsum(1).clamp_max(1).double().sum(0)
    if cmc.numel() < max_rank:
        cmc = torch.cat((cmc, cmc[-1].expand(max_rank - cmc.numel())))
    if not full:
        return cmc, None, None
    positions = torch.arange(1, matches.shape[1] + 1, dtype=torch.float64)[None]
    num_rel = matches.sum(1).double()
    precision = matches.cumsum(1).double() / positions
    ap = (precision * matches).sum(1) / num_rel.clamp_min(1)
    last = (matches * positions).amax(1)
    inp = num_rel / last.clamp_min(1)
    return cmc, ap.sum(), inp.sum()


@torch.no_grad()
def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    """Historical full-matrix API, retained for analysis and regression tests."""
    _validate(q_pids, g_pids, max_rank)
    if similarity.shape != (q_pids.numel(), g_pids.numel()):
        raise ValueError("Similarity dimensions do not match identity arrays")
    indices = (similarity.argsort(dim=1, descending=True) if get_mAP else
               similarity.topk(min(max_rank, g_pids.numel()), dim=1).indices)
    cmc, ap, inp = _statistics(indices, q_pids, g_pids, max_rank, get_mAP)
    scale = 100.0 / q_pids.numel()
    if not get_mAP:
        return (cmc * scale).float(), indices
    return (cmc * scale).float(), (ap * scale).float(), (inp * scale).float(), indices


def _rank_chunked(qfeats, gfeats, q_pids, g_pids, max_rank, chunk_size, full):
    _validate(q_pids, g_pids, max_rank)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if (qfeats.ndim != 2 or gfeats.ndim != 2 or
            qfeats.shape[1] != gfeats.shape[1] or
            qfeats.shape[0] != q_pids.numel() or gfeats.shape[0] != g_pids.numel()):
        raise ValueError("Feature dimensions do not match query/gallery identities")
    cmc_sum = torch.zeros(max_rank, dtype=torch.float64)
    ap_sum = inp_sum = 0.0
    for start in range(0, len(qfeats), chunk_size):
        sim = qfeats[start:start + chunk_size] @ gfeats.t()
        indices = (sim.argsort(dim=1, descending=True) if full else
                   sim.topk(min(max_rank, len(gfeats)), dim=1).indices)
        cmc, ap, inp = _statistics(indices, q_pids[start:start + chunk_size],
                                   g_pids, max_rank, full)
        cmc_sum += cmc
        if full:
            ap_sum += ap.item()
            inp_sum += inp.item()
    scale = 100.0 / len(qfeats)
    cmc = (cmc_sum * scale).float()
    if not full:
        return cmc
    return cmc, torch.tensor(ap_sum * scale), torch.tensor(inp_sum * scale)


@torch.no_grad()
def rank_topk_chunked(qfeats, gfeats, q_pids, g_pids, max_rank=10, chunk_size=256):
    return _rank_chunked(qfeats, gfeats, q_pids, g_pids, max_rank, chunk_size, False)


@torch.no_grad()
def rank_full_chunked(qfeats, gfeats, q_pids, g_pids, max_rank=10, chunk_size=128):
    return _rank_chunked(qfeats, gfeats, q_pids, g_pids, max_rank, chunk_size, True)


class Evaluator:
    def __init__(self, img_loader, txt_loader, chunk_size=128):
        self.img_loader, self.txt_loader = img_loader, txt_loader
        self.chunk_size = chunk_size
        self.logger = logging.getLogger("IRRA.eval")
        self.last_metrics = {}

    @torch.no_grad()
    def _compute_embedding(self, model):
        model.eval()
        device = next(model.parameters()).device
        qids, gids, qfeats, gfeats = [], [], [], []
        for pid, caption in self.txt_loader:
            qids.append(pid.view(-1).cpu())
            qfeats.append(model.encode_text(caption.to(device, non_blocking=True)))
        for pid, image in self.img_loader:
            gids.append(pid.view(-1).cpu())
            gfeats.append(model.encode_image(image.to(device, non_blocking=True)))
        if not qfeats or not gfeats:
            raise ValueError("Empty evaluation split; select a populated split")
        return torch.cat(qfeats), torch.cat(gfeats), torch.cat(qids), torch.cat(gids)

    @torch.no_grad()
    def eval(self, model, i2t_metric=False, full_metrics=True):
        q, g, qi, gi = self._compute_embedding(model)
        q, g = F.normalize(q.float(), dim=1), F.normalize(g.float(), dim=1)
        columns = ["task", "R1", "R5", "R10"] + (["mAP", "mINP"] if full_metrics else [])
        table = PrettyTable(columns)
        self.last_metrics = {}
        directions = [("t2i", q, g, qi, gi)]
        if i2t_metric:
            directions.append(("i2t", g, q, gi, qi))
        for name, a, b, ai, bi in directions:
            if full_metrics:
                cmc, ap, inp = rank_full_chunked(a, b, ai, bi, chunk_size=self.chunk_size)
            else:
                cmc = rank_topk_chunked(a, b, ai, bi, chunk_size=self.chunk_size)
            values = [cmc[0].item(), cmc[4].item(), cmc[9].item()]
            if full_metrics:
                values += [ap.item(), inp.item()]
            metrics = dict(zip(columns[1:], values))
            metrics.update(num_queries=len(a), num_gallery=len(b),
                           missing_queries=int((~torch.isin(ai, bi)).sum()))
            self.last_metrics[name] = metrics
            table.add_row([name] + [f"{v:.3f}" for v in values])
        self.logger.info("\n%s", table)
        return self.last_metrics["t2i"]["R1"]
