from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/iira/configs.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--eval_chunk_size', type=int, default=None)
    cli = parser.parse_args()
    args = load_train_configs(cli.config_file)
    checkpoint = cli.checkpoint or op.join(args.output_dir, 'best.pth')
    for key in ('root_dir', 'output_dir', 'device', 'eval_chunk_size'):
        if getattr(cli, key) is not None:
            setattr(args, key, getattr(cli, key))

    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = getattr(args, 'device', 'cuda')
    args.device = device

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=checkpoint)
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader,
                 output_file=op.join(args.output_dir, 'metrics.json'),
                 chunk_size=getattr(args, 'eval_chunk_size', 128))
