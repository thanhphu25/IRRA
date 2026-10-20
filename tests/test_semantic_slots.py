import torch

from model.scfp.semantic_slots import DualModalSemanticSlots, text_content_mask
from model.scfp.diversity import slot_diversity


def test_shapes_content_mask_normalization_and_gradients():
    torch.manual_seed(42)
    m = DualModalSemanticSlots()
    image = torch.randn(2, 192, 512, requires_grad=True)
    text = torch.randn(2, 77, 512, requires_grad=True)
    ids = torch.zeros(2, 77, dtype=torch.long)
    ids[:, :5] = torch.tensor([49406, 11, 12, 49407, 99])
    f = m(image, text, ids)
    assert f['visual_slots'].shape == f['text_slots'].shape == (2, 4, 512)
    assert f['text_content_mask'].sum() == 4
    assert not f['text_attention'][:, :, [0, 3, 4]].any()
    for name in ('visual_slots', 'text_slots'):
        torch.testing.assert_close(f[name].norm(dim=-1), torch.ones(2, 4))
        assert slot_diversity(f[name]) < 0.999
    torch.testing.assert_close(f['slot_relevance'].sum(-1), torch.ones(2))
    assert (f['slot_relevance_raw'] >= 0.1).all()
    objective = (f['visual_slots'] * f['text_slots']).sum() + f['slot_relevance'][:, 0].sum()
    objective.backward()
    assert image.grad.abs().sum() and text.grad.abs().sum()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_padding_invariance_and_empty_content():
    m = DualModalSemanticSlots(dim=32, heads=4).eval()
    ids = torch.tensor([[49406, 5, 49407, 0, 0], [49406, 49407, 0, 0, 0]])
    i, t = torch.randn(2, 9, 32), torch.randn(2, 5, 32)
    f = m(i, t, ids)
    t2 = t.clone()
    t2[~text_content_mask(ids)] = 1e5
    f2 = m(i, t2, ids)
    torch.testing.assert_close(f['text_slots'], f2['text_slots'])
    assert torch.isfinite(f['text_slots']).all()
    assert not f['text_attention'][1].any()


def test_single_slot_diversity_is_differentiable_zero():
    x = torch.randn(2, 1, 32, requires_grad=True)
    loss = slot_diversity(x)
    assert loss == 0
    loss.backward()
    assert x.grad is not None
