import pytest
import torch
import torch.nn.functional as F

from model.scfp.local_alignment import positive_targets, local_similarity, local_alignment_loss
from model.objectives import compute_sdm


@pytest.mark.parametrize('pids', [[1, 1, 2, 3], [1], [1, 1, 1], [1, 2]])
def test_multi_positive_bidirectional_sdm(pids):
    ids = torch.tensor(pids)
    b = len(pids)
    v, t = torch.randn(b, 4, 32, requires_grad=True), torch.randn(b, 4, 32, requires_grad=True)
    rel = torch.ones(b, 4, requires_grad=True)
    targets = positive_targets(ids)
    assert targets[0, 0] == (0.5 if pids == [1, 1, 2, 3] else 1 / pids.count(pids[0]))
    similarity = local_similarity(v, t, rel)
    assert similarity.shape == (b, b)
    loss = local_alignment_loss(similarity, ids, 10.0)
    torch.testing.assert_close(loss, local_alignment_loss(similarity.t(), ids, 10.0))
    assert torch.isfinite(loss)
    loss.backward()
    for x in (v, t, rel):
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_one_slot_agrees_with_original_global_sdm():
    v, t, ids = torch.randn(4, 32), torch.randn(4, 32), torch.tensor([1, 1, 2, 3])
    similarity = F.normalize(v, dim=-1) @ F.normalize(t, dim=-1).t()
    torch.testing.assert_close(local_alignment_loss(similarity, ids, 10), compute_sdm(v, t, ids, 10))
