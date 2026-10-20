import pytest
import torch

from solver import build_optimizer, build_lr_scheduler
from utils.checkpoint import Checkpointer


def test_baseline_warm_start_fresh_optimizer(make_model, tmp_path):
    baseline, model = make_model(), make_model('frequency')
    old_optimizer = build_optimizer(baseline.args, baseline)
    for p in baseline.parameters():
        p.grad = torch.ones_like(p)
    old_optimizer.step()
    path = tmp_path / 'old.pth'
    torch.save({'model': {'module.' + k: v for k, v in baseline.state_dict().items()},
                'optimizer': old_optimizer.state_dict(), 'epoch': 60}, path)
    optimizer = build_optimizer(model.args, model)
    report = Checkpointer(model, optimizer).warm_start(path)
    assert report['missing_keys'] == ['spatial_prompt.frequency.alpha', 'spatial_prompt.frequency.logits']
    assert not optimizer.state
    for k, v in baseline.state_dict().items():
        assert torch.equal(v, model.state_dict()[k])
    with pytest.raises(RuntimeError, match='Missing'):
        Checkpointer(make_model('frequency')).load(path)


@pytest.mark.parametrize('mode', ['baseline', 'frequency', 'local', 'both'])
def test_new_checkpoint_roundtrip_resume(make_model, batch, tmp_path, mode):
    m = make_model(mode)
    opt = build_optimizer(m.args, m)
    sched = build_lr_scheduler(m.args, opt)
    ret = m(batch)
    sum(v for k, v in ret.items() if 'loss' in k).backward()
    opt.step()
    sched.step()
    Checkpointer(m, opt, sched, str(tmp_path), True).save('best', epoch=2, best_top1=7.0)
    clone = make_model(mode)
    copt = build_optimizer(clone.args, clone)
    csched = build_lr_scheduler(clone.args, copt)
    meta = Checkpointer(clone, copt, csched).resume(tmp_path / 'best.pth')
    assert meta['epoch'] == 2 and meta['best_top1'] == 7.0
    assert copt.state and csched.state_dict() == sched.state_dict()
    m.eval(); clone.eval()
    torch.testing.assert_close(m.encode_image(batch['images']), clone.encode_image(batch['images']), rtol=0, atol=0)


def test_reject_non_tensor_prompt_config_mismatch(make_model, tmp_path):
    m = make_model('frequency')
    Checkpointer(m, save_dir=str(tmp_path), save_to_disk=True).save('best')
    altered = make_model('frequency', frequency_temperature=2.0)
    with pytest.raises(RuntimeError, match='configuration differs'):
        Checkpointer(altered).load(tmp_path / 'best.pth')


@pytest.mark.parametrize('damage', ['missing', 'unexpected', 'shape', 'prompt', 'all_prompts'])
def test_reject_corruption(make_model, tmp_path, damage):
    m = make_model('frequency')
    state = m.state_dict()
    if damage == 'missing':
        state.pop('classifier.bias')
    elif damage == 'unexpected':
        state['unknown'] = torch.ones(1)
    elif damage == 'shape':
        state['classifier.bias'] = torch.zeros(5)
    elif damage == 'prompt':
        state.pop('spatial_prompt.frequency.alpha')
    else:
        state = {k: v for k, v in state.items() if not k.startswith('spatial_prompt.')}
    path = tmp_path / 'broken.pth'
    torch.save({'model': state, 'model_config': vars(m.args)}, path)
    with pytest.raises(RuntimeError):
        Checkpointer(m).warm_start(path)
