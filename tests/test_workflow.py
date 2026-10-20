import importlib
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from tests.helpers import tiny_clip, tiny_args, synthetic_batch
from datasets.splits import train_validation_holdout
from experiments.scfp.prepare_kaggle import prepare_icfg, find_checkpoint
from utils.options import get_args
from utils.checkpoint import Checkpointer
from solver import build_optimizer, build_lr_scheduler
from processor.processor import do_train


def test_protocols_and_cli():
    assert get_args([]).scfp_mode == 'baseline'
    assert get_args(['--img_size', '256', '128']).img_size == [256, 128]
    for argv in (['--scfp_mode', 'v2'], ['--training_protocol', 'warmstart'],
                 ['--training_protocol', 'scientific'],
                 ['--init_checkpoint', 'x', '--resume_checkpoint', 'y']):
        with pytest.raises(SystemExit):
            get_args(argv)
    args = get_args(['--training_protocol', 'warmstart', '--val_dataset', 'none', '--init_checkpoint', 'x'])
    assert args.MLM


def test_identity_holdout_fixed_disjoint():
    rows = [(pid, pid * 2 + i, str(pid * 2 + i), 'caption') for pid in range(10) for i in range(2)]
    a, val, ids = train_validation_holdout(rows, seed=42)
    assert not set(r[0] for r in a) & set(ids)
    assert len(a) + len(val['captions']) == len(rows)
    assert (a, val, ids) == train_validation_holdout(rows, seed=42)


def test_kaggle_preparation_and_ambiguous_checkpoint(tmp_path):
    root = tmp_path / 'input'
    dataset = root / 'dataset'
    (dataset / 'imgs').mkdir(parents=True)
    (dataset / 'imgs/a.jpg').touch()
    annotation = dataset / 'ICFG-PEDES.json'
    annotation.write_text(json.dumps([dict(split='train', file_path='a.jpg', id=0, captions=['a'])]))
    output = tmp_path / 'data'
    prepare_icfg(root, output)
    prepare_icfg(root, output)
    assert (output / 'ICFG-PEDES/imgs/a.jpg').is_file()
    for name in ['a', 'b']:
        (root / name).mkdir()
        (root / name / 'best.pth').touch()
    with pytest.raises(ValueError, match='Expected one'):
        find_checkpoint(root)


def test_training_logs_checkpoint_no_test_selection_and_resume(monkeypatch, tmp_path):
    build = importlib.import_module('model.build')
    monkeypatch.setattr(build, 'build_CLIP_from_openai_pretrained', tiny_clip)
    args = tiny_args()
    args.output_dir = str(tmp_path)
    args.num_epoch = 2
    args.distributed = False
    args.val_dataset = 'none'
    args.training_protocol = 'warmstart'
    args.scfp_diagnostics = True
    args.max_steps = 1
    model = build.build_model(args, 3)
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    cp = Checkpointer(model, optimizer, scheduler, str(tmp_path), True)
    batch = synthetic_batch()
    loader = DataLoader([{k: v[i] for k, v in batch.items()} for i in range(4)], batch_size=4)
    do_train(1, args, model, loader, None, optimizer, scheduler, cp)
    for name in ['last.pth', 'training_stats.jsonl', 'validation_metrics.jsonl', 'metrics.json',
                 'diagnostics.json', 'fixed_sample_features.pt']:
        assert (tmp_path / name).is_file()
    assert not (tmp_path / 'best.pth').exists()
    assert json.loads((tmp_path / 'metrics.json').read_text())['official_test'] == 'NOT RUN'
    state = cp.resume(tmp_path / 'last.pth')
    assert state['epoch'] == 2 and state['best_top1'] is None


def test_yaml_ablation_presets_and_cli_override():
    for mode in ('baseline', 'slots', 'frequency', 'v2'):
        args = get_args(['--experiment_config', f'experiments/scfp/configs/{mode}.yaml',
                         '--training_protocol', 'scientific', '--val_dataset', 'holdout',
                         '--scfp_num_slots', '3'])
        assert args.scfp_mode == mode and args.scfp_num_slots == 3


def test_saved_config_is_resume_cli_input(tmp_path):
    from utils.iotools import save_train_configs
    args = tiny_args()
    args.distributed = False
    args.training_protocol = 'warmstart'
    args.val_dataset = 'none'
    args.init_checkpoint = '/original/best.pth'
    save_train_configs(tmp_path, args)
    resumed = get_args(['--experiment_config', str(tmp_path / 'configs.yaml'),
                        '--init_checkpoint', '', '--resume_checkpoint', '/new/last.pth'])
    assert resumed.resume_checkpoint == '/new/last.pth'
    assert resumed.scfp_mode == 'v2' and resumed.init_checkpoint == ''
