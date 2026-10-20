import importlib
import random

import numpy as np
import pytest
import torch

from model.clip_model import convert_weights
from tests.helpers import tiny_clip, tiny_args, synthetic_batch
from utils.checkpoint import Checkpointer, NEW_PREFIXES, load_state_dict
from solver import build_optimizer, build_lr_scheduler


@pytest.fixture
def builder(monkeypatch):
    build = importlib.import_module('model.build')
    monkeypatch.setattr(build, 'build_CLIP_from_openai_pretrained', tiny_clip)
    return build.build_model


def test_original_main_regression(builder, monkeypatch):
    from pathlib import Path
    import types
    reference = types.ModuleType('model.irra_reference')
    reference.__package__ = 'model'
    source = Path(__file__).parent / 'fixtures/irra_build_main.py'
    exec(compile(source.read_text(), str(source), 'exec'), reference.__dict__)
    monkeypatch.setattr(reference, 'build_CLIP_from_openai_pretrained', tiny_clip)
    args = tiny_args('baseline')
    args.precision = 'fp16'
    original = reference.build_model(args, 3).eval()
    current = builder(args, 3).eval()
    assert set(original.state_dict()) == set(current.state_dict())
    current.load_state_dict(original.state_dict())
    batch = synthetic_batch()
    with torch.no_grad():
        for k, v in original(batch).items():
            torch.testing.assert_close(current(batch)[k], v)
        torch.testing.assert_close(current.encode_image(batch['images']), original.encode_image(batch['images']))
        torch.testing.assert_close(current.encode_text(batch['caption_ids']), original.encode_text(batch['caption_ids']))


def test_warm_start_only_new_missing_no_optimizer_restore(builder, tmp_path):
    baseline = builder(tiny_args('baseline'), 3)
    optimizer = build_optimizer(baseline.args, baseline)
    sum(v for k, v in baseline(synthetic_batch()).items() if k.endswith('_loss')).backward()
    optimizer.step()
    Checkpointer(baseline, optimizer, save_dir=str(tmp_path), save_to_disk=True).save('old', epoch=60)
    v2 = builder(tiny_args(), 3)
    new_optimizer = build_optimizer(v2.args, v2)
    cp = Checkpointer(v2, new_optimizer)
    cp.load(tmp_path / 'old.pth', warm_start=True)
    assert new_optimizer.state == {}
    assert cp.last_report['missing_keys']
    assert all(k.startswith(NEW_PREFIXES) for k in cp.last_report['missing_keys'])
    assert not cp.last_report['unexpected_keys']
    for k, v in baseline.state_dict().items():
        torch.testing.assert_close(v2.state_dict()[k], v)


def test_strict_corruption_prefix_and_partial_v2_rejected(builder):
    model = builder(tiny_args(), 3)
    state = model.state_dict()
    load_state_dict(model, {'module.' + k: v for k, v in state.items()})
    for bad in ({k: v for k, v in state.items() if k != 'scfp_slots.shared_queries'},
                dict(state, unexpected=torch.ones(1)),
                dict(state, **{'classifier.weight': torch.zeros(2, 512)})):
        with pytest.raises(RuntimeError, match='Incompatible'):
            load_state_dict(model, bad, allow_new=True)
    baseline = builder(tiny_args('baseline'), 3)
    bad = baseline.state_dict()
    del bad['base_model.visual.proj']
    with pytest.raises(RuntimeError):
        load_state_dict(model, bad, allow_new=True)


def test_resume_optimizer_scheduler_rng_and_architecture(builder, tmp_path):
    model = builder(tiny_args(), 3)
    optimizer = build_optimizer(model.args, model)
    scheduler = build_lr_scheduler(model.args, optimizer)
    sum(v for k, v in model(synthetic_batch()).items() if k.endswith('_loss')).backward()
    optimizer.step()
    scheduler.step()
    cp = Checkpointer(model, optimizer, scheduler, str(tmp_path), True)
    cp.save('last', epoch=3, best_top1=0.0, best_epoch=1)
    expected = (torch.rand(1), random.random(), np.random.rand())
    metadata = cp.resume(tmp_path / 'last.pth')
    observed = (torch.rand(1), random.random(), np.random.rand())
    assert metadata['epoch'] + 1 == 4 and metadata['best_top1'] == 0.0
    torch.testing.assert_close(expected[0], observed[0])
    assert expected[1:] == observed[1:]
    assert optimizer.state and scheduler.last_epoch == 1
    model.args.scfp_frequency_layer = 2
    with pytest.raises(RuntimeError, match='scfp_frequency_layer'):
        cp.load(tmp_path / 'last.pth')


def test_offline_backbone_construction_and_all_new_keys_removed(builder, tmp_path):
    baseline = builder(tiny_args('baseline'), 3)
    Checkpointer(baseline, save_dir=str(tmp_path), save_to_disk=True).save('baseline')
    args = tiny_args()
    args.init_checkpoint = str(tmp_path / 'baseline.pth')
    offline = builder(args, 3)
    cp = Checkpointer(offline)
    cp.load(args.init_checkpoint, warm_start=True)
    assert offline.encode_image(synthetic_batch()['images']).shape == (4, 512)
    Checkpointer(offline, save_dir=str(tmp_path), save_to_disk=True).save('v2')
    path = tmp_path / 'v2.pth'
    checkpoint = torch.load(path, weights_only=False)
    checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if not k.startswith(NEW_PREFIXES)}
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match='Incompatible'):
        cp.load(path, warm_start=True)
