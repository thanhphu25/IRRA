import torch
import torch.nn.functional as F
import pytest

from solver import build_optimizer


@pytest.mark.parametrize('mode', ['baseline', 'frequency', 'local', 'both'])
def test_train_inference_losses_optimizer(make_model, batch, mode):
    model = make_model(mode)
    seen = []
    h = model.base_model.visual.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
    ret = model(batch)
    assert {'sdm_loss', 'mlm_loss', 'id_loss', 'img_acc', 'txt_acc', 'mlm_acc', 'temperature'} == set(ret)
    loss = sum(v for k, v in ret.items() if 'loss' in k)
    assert loss.isfinite()
    loss.backward()
    optimizer = build_optimizer(model.args, model)
    params = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    for name, p in model.spatial_prompt.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0, name
        assert params[id(p)] == model.args.lgsp_lr
    model.eval()
    image = model.encode_image(batch['images'])
    text = model.encode_text(batch['caption_ids'])
    assert image.shape == text.shape == (3, 64)
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
    h.remove()
    if mode == 'baseline':
        baseline = model.base_model.encode_image(batch['images'])[:, 0].float()
        torch.testing.assert_close(image, baseline, rtol=0, atol=0)


def test_near_identity_embedding_and_text(make_model, batch):
    baseline, prompted = make_model(), make_model('frequency')
    baseline.eval(); prompted.eval()
    i0, i1 = baseline.encode_image(batch['images']), prompted.encode_image(batch['images'])
    cosine = F.cosine_similarity(i0, i1).mean().item()
    difference = (i0 - i1).abs().max().item()
    print(f'near_identity cosine={cosine:.9f} max_embedding_difference={difference:.9g}')
    assert cosine > 0.9999 and difference < 0.01
    torch.testing.assert_close(baseline.encode_text(batch['caption_ids']),
                               prompted.encode_text(batch['caption_ids']), rtol=0, atol=0)


def test_frozen_strategy(make_model, batch):
    model = make_model('frequency', freeze_backbone=True)
    assert all(p.requires_grad == n.startswith('spatial_prompt.') for n, p in model.named_parameters())
    ret = model(batch)
    sum(v for k, v in ret.items() if 'loss' in k).backward()
    assert model.spatial_prompt.frequency.logits.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith('spatial_prompt.'))
