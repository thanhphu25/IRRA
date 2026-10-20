import ast
import io
import json
from pathlib import Path
import subprocess
import signal
import sys
from types import SimpleNamespace

import nbformat
from PIL import Image
import pytest
import torch
import yaml

from experiments.scfp import kaggle_train
from model.clip_model import CLIP
from utils.options import get_args


def test_irra_comparison_guards_and_baseline_settings():
    args = get_args(['--training_protocol', 'irra', '--scfp_mode', 'v2'])
    assert args.val_dataset == 'test' and not args.init_checkpoint
    for extra in (['--init_checkpoint', 'best.pth'], ['--val_dataset', 'holdout']):
        with pytest.raises(SystemExit):
            get_args(['--training_protocol', 'irra', '--scfp_mode', 'v2', *extra])
    original = dict(optimizer='Adam', lr_factor=5.0, bias_lr_factor=2.0, img_size=(384, 128),
                    loss_names='sdm+mlm+id', img_aug=True, warmup_epochs=5,
                    warmup_factor=0.1, lrscheduler='cosine', weight_decay=4e-5,
                    seed=7, init_checkpoint='old/best.pth', lgsp_mode='old', scfp_mode='frequency')
    settings = kaggle_train.comparison_settings(original)
    for key in ('optimizer', 'lr_factor', 'bias_lr_factor', 'img_size', 'loss_names', 'img_aug',
                'warmup_epochs', 'warmup_factor', 'lrscheduler', 'weight_decay'):
        assert settings[key] == original[key]
    assert settings['num_epoch'] == 60 and settings['batch_size'] == 64 and settings['lr'] == 1e-5
    assert settings['training_protocol'] == 'irra' and settings['seed'] == 7
    assert settings['scfp_mode'] == 'v2' and settings['max_steps'] == 0
    assert settings['init_checkpoint'] == '' and 'lgsp_mode' not in settings
    assert kaggle_train.comparison_settings({})['seed'] == 1
    assert kaggle_train.comparison_settings({'seed': None})['seed'] == 1


def test_lean_notebook_schema_and_from_clip_defaults():
    notebook = nbformat.read('notebooks/irra_scfp_v2_train_kaggle.ipynb', as_version=4)
    nbformat.validate(notebook)
    code = [c.source for c in notebook.cells if c.cell_type == 'code']
    assert len(code) == 3
    for source in code:
        ast.parse(source)
    config = {}
    exec(code[0], config)
    assert config['EPOCHS'] == 60 and config['BATCH_SIZE'] == 64 and config['EXPERIMENT'] == 'v2'
    assert config['BACKBONE_LR'] == 1e-5 and config['RESUME_CHECKPOINT'] is None
    assert 'INIT_CHECKPOINT' not in config and 'BASELINE_CONFIG' not in config and 'SOURCE' not in config
    assert config['BRANCH'] == 'feat/scfp-v2' and config['SEED'] == 1
    assert not any('pytest' in c or 'smoke_test' in c for c in code)


@pytest.mark.parametrize('published', [True, False])
def test_notebook_clones_requested_branch_and_detects_missing_launcher(tmp_path, published):
    import tempfile
    source = tmp_path / 'git_source'
    source.mkdir()
    (source / 'README.md').write_text('local clone fixture')
    if published:
        launcher = source / 'experiments/scfp/kaggle_train.py'
        launcher.parent.mkdir(parents=True)
        launcher.touch()
        (launcher.parent / 'requirements-train.txt').touch()
    def git(*command):
        return subprocess.run(['git', '-c', 'user.name=Notebook Test',
                               '-c', 'user.email=notebook@example.invalid',
                               '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
                               *command], cwd=source, check=True, capture_output=True, text=True)
    git('init', '-b', 'feat/scfp-v2')
    git('add', '.')
    git('commit', '-m', 'Local notebook clone fixture')
    notebook = nbformat.read('notebooks/irra_scfp_v2_train_kaggle.ipynb', as_version=4)
    setup = notebook.cells[2].source
    clone = setup[setup.index('REPO = '):setup.index('os.chdir(REPO)')]
    env = dict(Path=Path, subprocess=subprocess, BRANCH='feat/scfp-v2', REPO_URL=str(source),
               tempfile=SimpleNamespace(mkdtemp=lambda **kw: tempfile.mkdtemp(prefix=kw['prefix'], dir=tmp_path)))
    if not published:
        with pytest.raises(RuntimeError, match='push'):
            exec(clone, env)
    else:
        exec(clone, env)
        result = subprocess.check_output(['git', '-C', str(env['REPO']), 'branch', '--show-current'], text=True)
        assert result.strip() == 'feat/scfp-v2'
        assert (env['REPO'] / 'experiments/scfp/kaggle_train.py').is_file()


def test_prepare_from_dataset_only_without_original_config(tmp_path):
    dataset, unused_config = _toy_input(tmp_path)
    unused_config.unlink()
    working = tmp_path / 'working'
    plan = kaggle_train.prepare_training(dataset_dir=dataset, output_root=working / 'runs',
                                         working_root=working, num_workers=0)
    preset = Path(plan['command'][plan['command'].index('--experiment_config') + 1])
    settings = yaml.safe_load(preset.read_text())
    assert settings['pretrain_choice'] == 'ViT-B/16' and settings['img_size'] == [384, 128]
    assert settings['optimizer'] == 'Adam' and settings['lrscheduler'] == 'cosine'
    assert settings['warmup_epochs'] == 5 and settings['cmt_depth'] == 4
    assert settings['num_epoch'] == 60 and settings['batch_size'] == 64 and settings['seed'] == 1
    assert settings['img_aug'] and settings['MLM'] and settings['loss_names'] == 'sdm+mlm+id'
    assert settings['init_checkpoint'] == '' and settings['scfp_mode'] == 'v2'
    assert not (preset.parent / 'baseline_config.yaml').exists()


def test_notebook_rerun_evicts_previous_source_modules_only(tmp_path):
    notebook = nbformat.read('notebooks/irra_scfp_v2_train_kaggle.ipynb', as_version=4)
    setup = notebook.cells[2].source
    start = setup.index('cached_launcher = ')
    cache_reset = setup[start:setup.index('subprocess.run(', start)]
    old = tmp_path / 'previous_source'
    launcher = SimpleNamespace(__file__=str(old / 'experiments/scfp/kaggle_train.py'))
    modules = {'experiments.scfp.kaggle_train': launcher,
               'utils.options': SimpleNamespace(__file__=str(old / 'utils/options.py')),
               'experiments': SimpleNamespace(), 'experiments.scfp': SimpleNamespace(),
               'unrelated': SimpleNamespace(__file__=str(tmp_path / 'other_package.py'))}
    exec(cache_reset, dict(Path=Path, sys=SimpleNamespace(modules=modules)))
    assert set(modules) == {'unrelated'}


def _toy_input(root):
    dataset = root / 'input/dataset/ICFG-PEDES'
    (dataset / 'imgs').mkdir(parents=True)
    records = []
    for pid in range(6):
        name = f'{pid}.jpg'
        Image.new('RGB', (16, 32), (pid * 30, 40, 80)).save(dataset / 'imgs' / name)
        records.append(dict(id=pid, split='train' if pid < 4 else 'test', file_path=name,
                            captions=['a man wearing blue jeans']))
    (dataset / 'ICFG-PEDES.json').write_text(json.dumps(records))
    clip = CLIP(embed_dim=64, image_resolution=(32, 16), vision_layers=3,
                vision_width=64, vision_patch_size=8, stride_size=8,
                context_length=8, vocab_size=49408, transformer_width=64,
                transformer_heads=1, transformer_layers=1)
    pretrained = root / 'toy_clip.pt'
    torch.save(clip.state_dict(), pretrained)
    config = root / 'baseline.yaml'
    config.write_text(yaml.safe_dump(dict(pretrain_choice=str(pretrained), img_size=[32, 16],
                                         stride_size=8, text_length=8, cmt_depth=1, img_aug=True,
                                         loss_names='sdm+mlm+id', num_epoch=60, warmup_epochs=0)))
    return dataset, config


def test_lean_train_full_toy_epoch_evaluate_and_resume_from_input(monkeypatch, tmp_path):
    dataset, baseline = _toy_input(tmp_path)
    working = tmp_path / 'working'
    plan = kaggle_train.prepare_training(dataset_dir=dataset, baseline_config=baseline,
                                         output_root=working / 'runs', working_root=working,
                                         epochs=1, batch_size=4, num_workers=0)
    real_run = kaggle_train._run_process
    def cpu_run(command):
        # Execute the actual CLI pipeline on CPU with a small, random CLIP.
        command = list(command)
        if Path(command[2]).name == 'train.py':
            command += ['--device', 'cpu', '--precision', 'fp32', '--scfp_frequency_layer', '1']
        elif Path(command[2]).name == 'test.py':
            command += ['--device', 'cpu', '--test_batch_size', '4']
        return real_run(command)
    monkeypatch.setenv('OMP_NUM_THREADS', '2')
    monkeypatch.setattr(kaggle_train, '_run_process', cpu_run)
    run = kaggle_train.run_training(plan)
    assert (run / 'best.pth').is_file() and (run / 'last.pth').is_file()
    assert not (run / 'split_manifest.json').exists()
    results = list(run.glob('best_test_metrics_*.json'))
    assert len(results) == 1 and 'mAP' in json.loads(results[0].read_text())['metrics']['t2i']
    summary = json.loads((run / 'metrics.json').read_text())
    assert summary['selection_split'] == 'test'
    assert summary['official_test'] == 'MONITORED_DURING_TRAINING'
    saved = yaml.safe_load((run / 'configs.yaml').read_text())
    assert not saved['init_checkpoint'] and saved['num_epoch'] == 1 and saved['max_steps'] == 0
    # Simulate attaching an old run as Kaggle Input (read-only in real Kaggle).
    import shutil
    mounted = tmp_path / 'input/old_run'
    shutil.copytree(run, mounted)
    (mounted / 'train_log.txt').chmod(0o444)
    before = (mounted / 'train_log.txt').read_bytes()
    resumed = kaggle_train.prepare_training(dataset_dir=dataset, baseline_config='/not/needed.yaml',
                                            output_root=working / 'runs', working_root=working,
                                            resume_checkpoint=mounted / 'last.pth', num_workers=0)
    assert Path(resumed['run_dir']).is_relative_to(working)
    assert Path(resumed['run_dir'], 'best.pth').is_file()
    assert Path(resumed['run_dir'], 'train_log.txt').stat().st_mode & 0o200
    assert kaggle_train.run_training(resumed, evaluate=False) == Path(resumed['run_dir'])
    assert (mounted / 'train_log.txt').read_bytes() == before
    assert not (mounted / 'train_log.txt').stat().st_mode & 0o200
    # A separately attached last.pth can restore its embedded config. Unrelated
    # datasets and arbitrary large files beside it must not be copied to Working.
    single = tmp_path / 'input/last_only'
    (single / 'imgs').mkdir(parents=True)
    (single / 'imgs/unrelated.jpg').write_bytes(b'not a run artifact')
    (single / 'unrelated.pt').write_bytes(b'do not stage')
    shutil.copy2(run / 'last.pth', single / 'last.pth')
    standalone = kaggle_train.prepare_training(dataset_dir=dataset, baseline_config='/not/needed.yaml',
                                                output_root=working / 'runs', working_root=working,
                                                resume_checkpoint=single / 'last.pth', num_workers=0)
    restored = Path(standalone['run_dir'])
    assert (restored / 'configs.yaml').is_file() and not (single / 'configs.yaml').exists()
    assert not (restored / 'imgs').exists() and not (restored / 'unrelated.pt').exists()
    kaggle_train.run_training(standalone)
    metrics_file = next(restored.glob('last_test_metrics_*.json'))
    assert Path(json.loads(metrics_file.read_text())['checkpoint']).name == 'last.pth'


@pytest.mark.parametrize('force_kill', [False, True])
def test_interrupted_process_stops_its_worker_group(monkeypatch, force_kill):
    calls, launches = [], []
    class Process:
        pid = 123456
        stdout = io.StringIO('Epoch 1\n')
        def poll(self):
            return None
        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            if force_kill and timeout == 10:
                raise subprocess.TimeoutExpired('train', timeout)
            return -signal.SIGTERM
    def launch(command, **kwargs):
        launches.append(kwargs)
        return Process()
    monkeypatch.setattr(kaggle_train.subprocess, 'Popen', launch)
    monkeypatch.setattr(kaggle_train.os, 'killpg', lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        kaggle_train._run_process(['training-command'])
    assert launches[0]['start_new_session'] and launches[0]['cwd'] == kaggle_train.REPO_ROOT
    expected = [(Process.pid, signal.SIGTERM)]
    if force_kill:
        expected.append((Process.pid, signal.SIGKILL))
    assert calls == expected


def test_process_failure_and_execution_from_another_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    kaggle_train._run_process([sys.executable, '-c',
                              'import pathlib,sys; assert pathlib.Path.cwd() == pathlib.Path(sys.argv[1])',
                              str(kaggle_train.REPO_ROOT)])
    with pytest.raises(subprocess.CalledProcessError) as error:
        kaggle_train._run_process([sys.executable, '-c', 'raise SystemExit(7)'])
    assert error.value.returncode == 7
