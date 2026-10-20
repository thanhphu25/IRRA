import pytest
import torch

from model.lgsp import LocalSpatialPrompt


@pytest.mark.parametrize('shape', [(2, 3, 384, 128), (2, 3, 224, 224), (1, 3, 33, 17)])
@pytest.mark.parametrize('selection', ['softmax', 'topk'])
def test_candidates_weights_gradients(shape, selection):
    torch.manual_seed(4)
    m = LocalSpatialPrompt(selection=selection)
    x = torch.randn(shape)
    candidates = m.candidates(x)
    assert len(candidates) == 4 and all(p.shape == x.shape for p in candidates)
    weights = m.selection_weights(x, candidates)
    assert weights.shape == (shape[0], 4, *shape[-2:])
    torch.testing.assert_close(weights.sum(1), torch.ones(shape[0], *shape[-2:]))
    if selection == 'topk':
        assert ((weights > 0).sum(1) == 2).all()
    y = m(x)
    assert y.shape == x.shape and y.isfinite().all()
    (y * torch.randn_like(y)).sum().backward()
    for p in m.parameters():
        assert p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0


def test_activation_checkpoint_matches_outputs_gradients():
    torch.manual_seed(6)
    regular, recompute = LocalSpatialPrompt(), LocalSpatialPrompt(checkpoint_candidates=True)
    recompute.load_state_dict(regular.state_dict())
    x = torch.randn(2, 3, 32, 16)
    a, b = regular(x), recompute(x)
    torch.testing.assert_close(a, b)
    a.square().mean().backward(); b.square().mean().backward()
    for p, q in zip(regular.parameters(), recompute.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_small_batch_saved_activation_budget():
    m = LocalSpatialPrompt()
    saved = []
    def pack(tensor):
        saved.append(tensor.numel() * tensor.element_size())
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        y = m(torch.randn(2, 3, 64, 32))
    assert sum(saved) < 16 * 1024**2
    print(f'local small-batch saved tensor upper bound: {sum(saved)} bytes')
    y.sum().backward()


def test_single_candidate_softmax_does_not_require_unused_topk():
    m = LocalSpatialPrompt(pool_size=1)
    x = torch.randn(1, 3, 16, 8)
    y, weights = m(x, return_weights=True)
    assert y.shape == x.shape and torch.equal(weights, torch.ones_like(weights))
