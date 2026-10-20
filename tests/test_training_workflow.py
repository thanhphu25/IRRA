import json
import pytest
import torch

from processor.processor import do_train, do_inference
from solver import build_optimizer, build_lr_scheduler
from utils.checkpoint import Checkpointer


@pytest.mark.parametrize('mode', ['baseline', 'frequency', 'local', 'both'])
def test_one_batch_training_and_final_metrics(make_model, batch, tmp_path, mode):
    model = make_model(mode, num_epoch=1, warmup_epochs=1, output_dir=str(tmp_path),
                       log_period=1, eval_period=3, lgsp_diagnostics=True)
    class Loader(list):
        batch_size = 3
    class Validation:
        last_metrics = {'t2i': {'R1': 0.0}}
        def eval(self, model, full_metrics=True):
            assert full_metrics is False
            return 0.0
    opt = build_optimizer(model.args, model)
    sched = build_lr_scheduler(model.args, opt)
    do_train(1, model.args, model, Loader([batch]), Validation(), opt, sched,
             Checkpointer(model, opt, sched, str(tmp_path), True))
    assert (tmp_path / 'best.pth').is_file() and (tmp_path / 'last.pth').is_file()
    assert json.loads((tmp_path / 'training_stats.jsonl').read_text())['epoch'] == 1
    assert sched.get_last_lr()[0] >= 0
    metrics = do_inference(model, [(batch['pids'], batch['images'])],
                           [(batch['pids'], batch['caption_ids'])],
                           output_file=str(tmp_path / 'metrics.json'), chunk_size=1)
    assert set(metrics['t2i']) >= {'R1', 'R5', 'R10', 'mAP', 'mINP'}
