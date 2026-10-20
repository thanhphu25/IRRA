import importlib

import pytest
import torch

from tests.helpers import tiny_clip, tiny_args, synthetic_batch
from solver.build import build_optimizer


@pytest.mark.parametrize('mode', ['baseline', 'slots', 'frequency', 'v2'])
def test_all_modes_train_inference_optimizer_roundtrip(monkeypatch, tmp_path, mode):
    build = importlib.import_module('model.build')
    monkeypatch.setattr(build, 'build_CLIP_from_openai_pretrained', tiny_clip)
    args, batch = tiny_args(mode), synthetic_batch()
    model = build.build_model(args, num_classes=3)
    ret = model(batch)
    assert {'sdm_loss', 'id_loss', 'mlm_loss'} <= ret.keys()
    assert ('slot_loss' in ret) == (mode in ('slots', 'v2'))
    sum(v for k, v in ret.items() if k.endswith('_loss')).backward()
    for name, p in model.named_parameters():
        if 'scfp_' in name:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    optimizer = build_optimizer(args, model)
    parameters = [p for g in optimizer.param_groups for p in g['params']]
    assert len({id(p) for p in parameters}) == len(parameters)
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters() if p.requires_grad}
    for (name, p), group in zip(model.named_parameters(), optimizer.param_groups):
        if 'scfp_' in name:
            assert group['lr'] == args.scfp_lr
    model.eval()
    with torch.no_grad():
        image = model.encode_image(batch['images'])
        text = model.encode_text(batch['caption_ids'])
        assert image.shape == text.shape == (4, 512)
        vi, vt = model.base_model(batch['images'], batch['caption_ids'])
        torch.testing.assert_close(image, vi[:, 0].float())
        torch.testing.assert_close(text, vt[torch.arange(4), batch['caption_ids'].argmax(-1)].float())
    from utils.checkpoint import Checkpointer
    cp = Checkpointer(model, optimizer, save_dir=str(tmp_path), save_to_disk=True)
    cp.save('last', epoch=1)
    other = build.build_model(args, num_classes=3).eval()
    Checkpointer(other).load(tmp_path / 'last.pth')
    torch.testing.assert_close(other.encode_image(batch['images']), image)
    torch.testing.assert_close(other.encode_text(batch['caption_ids']), text)


def test_frequency_embedding_near_identity(monkeypatch):
    build = importlib.import_module('model.build')
    monkeypatch.setattr(build, 'build_CLIP_from_openai_pretrained', tiny_clip)
    model = build.build_model(tiny_args('frequency'), 3).eval()
    x = synthetic_batch()['images']
    freq = model.base_model.visual.scfp_frequency
    with torch.no_grad():
        enhanced = model.encode_image(x)
        freq.enabled = False
        original = model.encode_image(x)
        freq.enabled = True
    cosine = torch.nn.functional.cosine_similarity(original, enhanced).mean().item()
    assert cosine > 0.9999
    assert (original - enhanced).abs().max() < 0.01


def test_frozen_backbone_zero_auxiliary_weights_and_feature_interface(monkeypatch):
    build = importlib.import_module('model.build')
    monkeypatch.setattr(build, 'build_CLIP_from_openai_pretrained', tiny_clip)
    args = tiny_args()
    args.freeze_backbone = True
    args.scfp_slot_loss_weight = args.scfp_diversity_weight = 0.0
    model = build.build_model(args, 3)
    ret, features = model(synthetic_batch(), return_features=True)
    assert ret['slot_loss'] == 0 and ret['diversity_loss'] == 0
    assert 'frequency_band_features' in features  # frozen input spectrum has no trainable ancestor
    for key in ('frequency_logits', 'frequency_weights',
                'visual_slots', 'text_slots', 'slot_relevance'):
        assert key in features and features[key].requires_grad
    sum(v for k, v in ret.items() if k.endswith('_loss')).backward()
    for name, p in model.base_model.named_parameters():
        if 'scfp_frequency.' not in name:
            assert not p.requires_grad and p.grad is None
    assert model.base_model.visual.scfp_frequency.gate.net[-1].weight.grad.abs().sum() > 0
    optimizer = build_optimizer(args, model)
    assert sum(len(g['params']) for g in optimizer.param_groups) == sum(p.requires_grad for p in model.parameters())
