"""Lean Kaggle launcher for matched, from-CLIP IRRA comparisons."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

from experiments.scfp.prepare_kaggle import prepare_icfg
from utils.iotools import load_train_configs
from utils.options import get_args


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ARTIFACTS = {
    'best.pth', 'last.pth', 'configs.yaml', 'environment.json', 'resume_environment.json',
    'train_log.txt', 'test_log.txt', 'metrics.json', 'training_stats.jsonl',
    'validation_metrics.jsonl', 'split_manifest.json', 'diagnostics.json',
    'fixed_diagnostics_batch.pt', 'fixed_sample_features.pt',
}


def _resume_config(checkpoint):
    config = checkpoint.with_name('configs.yaml')
    if config.is_file():
        return dict(load_train_configs(config))
    # User-owned V2 checkpoints contain their complete training configuration.
    # Only deserialize it when the separate YAML was not attached.
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    settings = saved.get('config') if isinstance(saved, dict) else None
    if not isinstance(settings, dict):
        raise ValueError('Resume needs configs.yaml beside the checkpoint or config metadata inside a V2 checkpoint')
    return settings


def _stage_resume(checkpoint, settings, output_root, experiment, working_root):
    if checkpoint.parent.is_relative_to(working_root):
        run_dir = checkpoint.parent
    else:
        run_dir = output_root / experiment / ('resume_' + _identifier())
        run_dir.mkdir(parents=True, exist_ok=False)
        # A checkpoint can be mounted at a dataset root alongside unrelated
        # images/code. Copy only its run artifacts, never the entire Input tree.
        for path in checkpoint.parent.iterdir():
            if path.is_file() and (path.name in RUN_ARTIFACTS or path.name == checkpoint.name
                                   or path.name.startswith('events.out.tfevents.')
                                   or ('_test_metrics_' in path.name and path.suffix == '.json')):
                destination = run_dir / path.name
                shutil.copy2(path, destination)
                # Input files may also have read-only permission bits. Only the
                # Working copy must be writable so logs can append on resume.
                destination.chmod(destination.stat().st_mode | 0o200)
    config = run_dir / 'configs.yaml'
    if not config.exists():
        with config.open('x') as stream:
            yaml.safe_dump(settings, stream, sort_keys=False)
    return run_dir / checkpoint.name


def _run_process(command):
    """Stream output, and stop the train process plus DataLoader workers on interrupt."""
    process = subprocess.Popen(command, cwd=REPO_ROOT,
                               env=dict(os.environ, PYTHONUNBUFFERED='1'),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace', bufsize=1,
                               start_new_session=True)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        raise
    finally:
        process.stdout.close()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


# Only original IRRA settings transfer; never an old experiment's model/weights.
BASELINE_FIELDS = (
    'pretrain_choice', 'img_size', 'stride_size', 'text_length', 'vocab_size',
    'cmt_depth', 'temperature', 'img_aug', 'MLM', 'loss_names',
    'id_loss_weight', 'mlm_loss_weight', 'optimizer', 'lr', 'lr_factor',
    'bias_lr_factor', 'momentum', 'weight_decay', 'weight_decay_bias', 'alpha', 'beta',
    'num_epoch', 'milestones', 'gamma', 'warmup_factor', 'warmup_epochs',
    'warmup_method', 'lrscheduler', 'target_lr', 'power', 'sampler', 'num_instance',
)


def comparison_settings(original=None, *, experiment='v2', epochs=60, batch_size=64,
                        seed=None, backbone_lr=1e-5, scfp_lr=1e-4, num_workers=4):
    original = original or {}
    if seed is None:
        seed = original.get('seed')
    if seed is None:
        seed = 1
    # Start from the repository's IRRA defaults. The original run config is an
    # optional override, not a required dataset/checkpoint input.
    defaults = vars(get_args([]))
    settings = {key: defaults[key] for key in BASELINE_FIELDS}
    settings.update({key: original[key] for key in BASELINE_FIELDS if key in original})
    settings.update(
        scfp_mode=experiment, training_protocol='irra', val_dataset='test',
        dataset_name='ICFG-PEDES', training=True, device='cuda', precision='fp16',
        num_epoch=epochs, batch_size=batch_size, test_batch_size=batch_size,
        seed=seed,
        lr=backbone_lr, scfp_lr=scfp_lr, num_workers=num_workers,
        # Match run_irra.sh if the old config omitted these fields.
        img_aug=original.get('img_aug', True), MLM=True,
        loss_names=original.get('loss_names', 'sdm+mlm+id'),
        init_checkpoint='', resume_checkpoint='', resume=False, resume_ckpt_file='',
        freeze_backbone=False, max_steps=0, eval_period=1, log_period=original.get('log_period', 100),
        scfp_diagnostics=False, name=f'{experiment}_irra_comparison',
    )
    return settings


def _identifier():
    return datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]


def prepare_training(*, dataset_dir, output_root, baseline_config=None, experiment='v2',
                     epochs=60, batch_size=64, seed=None, backbone_lr=1e-5,
                     scfp_lr=1e-4, num_workers=4, resume_checkpoint=None,
                     working_root='/kaggle/working'):
    output_root = Path(output_root).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    data_root = Path(working_root).resolve() / 'data'
    prepare_icfg(data_root=data_root, annotation_source=dataset_dir / 'ICFG-PEDES.json',
                 images_source=dataset_dir / 'imgs')
    output_root.mkdir(parents=True, exist_ok=True)
    if resume_checkpoint:
        checkpoint = Path(resume_checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        saved = _resume_config(checkpoint)
        if saved.get('training_protocol') != 'irra' or saved.get('max_steps', 0) != 0:
            raise ValueError('Resume requires a full-data IRRA-comparison run, not an engineering/holdout run')
        if saved.get('scfp_mode', 'baseline') != experiment:
            raise ValueError('EXPERIMENT must match the resumed run')
        checkpoint = _stage_resume(checkpoint, saved, output_root, experiment, Path(working_root).resolve())
        run_dir = checkpoint.parent
        command = [sys.executable, '-u', str(REPO_ROOT / 'train.py'), '--experiment_config', str(run_dir / 'configs.yaml'),
                   '--init_checkpoint', '', '--resume_checkpoint', str(checkpoint),
                   '--resume_ckpt_file', '', '--root_dir', str(data_root), '--num_workers', str(num_workers)]
        args = get_args(command[3:])
        plan = dict(command=command, run_dir=str(run_dir), data_root=str(data_root))
    else:
        original = load_train_configs(baseline_config) if baseline_config else {}
        settings = comparison_settings(original, experiment=experiment, epochs=epochs,
                                       batch_size=batch_size, seed=seed, backbone_lr=backbone_lr,
                                       scfp_lr=scfp_lr, num_workers=num_workers)
        launch_dir = output_root / experiment / _identifier()
        launch_dir.mkdir(parents=True, exist_ok=False)
        settings.update(root_dir=str(data_root), output_dir=str(launch_dir / 'runs'))
        preset = launch_dir / 'train.yaml'
        preset.write_text(yaml.safe_dump(settings, sort_keys=False))
        # Validate before spawning a process or downloading CLIP.
        args = get_args(['--experiment_config', str(preset)])
        if baseline_config:
            shutil.copy2(baseline_config, launch_dir / 'baseline_config.yaml')
        command = [sys.executable, '-u', str(REPO_ROOT / 'train.py'), '--experiment_config', str(preset)]
        plan = dict(command=command, run_parent=str(launch_dir / 'runs' / 'ICFG-PEDES'),
                    data_root=str(data_root))
    print(json.dumps(dict(mode=args.scfp_mode, initialization='CLIP' if not resume_checkpoint else 'resume',
                          epochs=args.num_epoch, batch_size=args.batch_size, seed=args.seed,
                          backbone_lr=args.lr, scfp_lr=args.scfp_lr, optimizer=args.optimizer,
                          scheduler=args.lrscheduler, image_size=args.img_size,
                          outputs=plan.get('run_dir', plan.get('run_parent')),
                          training='full official train split', selection='official-test R1 (original IRRA protocol)'),
                     indent=2))
    return plan


def run_training(plan, *, evaluate=True):
    print('Training outputs:', plan.get('run_dir', plan.get('run_parent')), flush=True)
    try:
        _run_process(plan['command'])
    except KeyboardInterrupt:
        print('Training stopped. Resume using the last.pth saved after the latest completed epoch.', flush=True)
        raise
    if 'run_dir' in plan:
        run_dir = Path(plan['run_dir'])
    else:
        candidates = list(Path(plan['run_parent']).glob('*/configs.yaml'))
        if len(candidates) != 1:
            raise RuntimeError(f'Expected exactly one new training run; found {candidates}')
        run_dir = candidates[0].parent
    if evaluate:
        # A resumed Input may contain last.pth only. Label the actual checkpoint
        # explicitly rather than claiming a missing historical best was evaluated.
        checkpoint = run_dir / 'best.pth'
        if not checkpoint.is_file():
            checkpoint = run_dir / 'last.pth'
        result = run_dir / f'{checkpoint.stem}_test_metrics_{_identifier()}.json'
        _run_process([sys.executable, '-u', str(REPO_ROOT / 'test.py'), '--config_file', str(run_dir / 'configs.yaml'),
                        '--checkpoint', str(checkpoint), '--root_dir', plan['data_root'],
                        '--metrics_output', str(result)])
        print('Evaluated checkpoint:', checkpoint)
        print(json.dumps(json.loads(result.read_text())['metrics'], indent=2))
    print('Run directory:', run_dir)
    print('Checkpoints/logs:', ', '.join(p.name for p in sorted(run_dir.iterdir()) if p.is_file()))
    return run_dir
