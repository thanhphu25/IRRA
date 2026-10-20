"""Exact global-embedding evaluation, reconstructing architecture from run config."""
import argparse
from pathlib import Path

import torch

from datasets import build_dataloader
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.logger import setup_logger
from utils.metrics import Evaluator
from experiments.scfp.diagnostics import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description='IRRA / SCFP exact official test')
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--root_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--metrics_output', required=True)
    cli = parser.parse_args(argv)
    args = load_train_configs(cli.config_file)
    args.training = False
    args.device = cli.device
    args.test_batch_size = cli.test_batch_size
    args.distributed = False
    if cli.root_dir:
        args.root_dir = cli.root_dir
    if cli.device == 'cpu':
        args.precision = 'fp32'
    output = Path(cli.metrics_output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    setup_logger('IRRA', str(output.parent), if_train=False)
    image_loader, text_loader, num_classes = build_dataloader(args)
    checkpoint = cli.checkpoint or str(Path(args.output_dir) / 'best.pth')
    args.backbone_checkpoint = checkpoint
    model = build_model(args, num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(checkpoint)
    model.to(cli.device)
    evaluator = Evaluator(image_loader, text_loader, chunk_size=cli.chunk_size)
    evaluator.eval(model.eval(), full_metrics=True)
    write_json(output, dict(checkpoint=checkpoint, config_file=cli.config_file,
                            scfp_mode=getattr(args, 'scfp_mode', 'baseline'), split='official_test',
                            checkpoint_report=checkpointer.last_report, metrics=evaluator.last_metrics))
    print(f'Metrics saved: {output}')


if __name__ == '__main__':
    main()
