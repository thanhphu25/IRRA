import pytest
import torch

from utils.metrics import rank, rank_full_chunked, rank_topk_chunked


def test_chunked_matches_original_formula():
    torch.manual_seed(3)
    q, g = torch.randn(13, 8), torch.randn(17, 8)
    qi, gi = torch.arange(13) % 4, torch.arange(17) % 4
    sim = q @ g.t()
    indices = sim.argsort(dim=1, descending=True)
    matches = gi[indices].eq(qi[:, None])
    cumulative = matches.cumsum(1)
    cmc = cumulative[:, :10].clamp_max(1).float().mean(0) * 100
    positions = torch.arange(1, len(g) + 1)
    ap = ((cumulative / positions) * matches).sum(1) / matches.sum(1)
    last = (positions * matches).amax(1)
    inp = matches.sum(1) / last
    expected = (cmc, ap.mean() * 100, inp.mean() * 100)
    for chunk in (1, 4, 30):
        actual = rank_full_chunked(q, g, qi, gi, chunk_size=chunk)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(rank_topk_chunked(q, g, qi, gi, chunk_size=chunk), cmc)
    for a, b in zip(rank(sim, qi, gi)[:3], expected):
        torch.testing.assert_close(a, b)


def test_missing_identity_small_gallery_percentages():
    q, g = torch.eye(3), torch.eye(3)[:2]
    qi, gi = torch.tensor([0, 1, 99]), torch.tensor([0, 1])
    cmc, ap, inp = rank_full_chunked(q, g, qi, gi)
    torch.testing.assert_close(cmc, torch.full((10,), 200 / 3))
    assert ap.item() == pytest.approx(200 / 3)
    assert inp.item() == pytest.approx(200 / 3)
    torch.testing.assert_close(rank_topk_chunked(q, g, qi, gi), cmc)


def test_no_relevant_images():
    cmc, ap, inp = rank_full_chunked(torch.ones(1, 2), torch.ones(3, 2),
                                    torch.tensor([9]), torch.arange(3))
    assert not cmc.any() and ap == 0 and inp == 0


def test_invalid_retrieval():
    with pytest.raises(ValueError, match="nonempty"):
        rank(torch.empty(0, 0), torch.empty(0), torch.empty(0))
