import os
import os.path as op
import torch
import numpy as np
import random
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize
from experiments.scfp.diagnostics import environment, write_json


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = get_args()
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.scfp_mode != 'baseline' and args.distributed:
        raise ValueError('SCFP V2 is currently validated for a single GPU only')

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    set_seed(args.seed + get_rank())
    
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu --precision fp32 for CPU checks')
    if args.resume_checkpoint:
        args.output_dir = str(Path(args.resume_checkpoint).resolve().parent)
    elif get_rank() == 0:
        cur_time = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d_%H%M%S_%f')
        args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}_{uuid.uuid4().hex[:6]}')
        os.makedirs(args.output_dir, exist_ok=False)
    if args.distributed:
        paths = [args.output_dir if get_rank() == 0 else None]
        torch.distributed.broadcast_object_list(paths, src=0)
        args.output_dir = paths[0]
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    if get_rank() == 0 and not args.resume_checkpoint:
        save_train_configs(args.output_dir, args)
        write_json(Path(args.output_dir) / 'environment.json', environment(args))
    elif get_rank() == 0:
        write_json(Path(args.output_dir) / 'resume_environment.json', environment(args))
    if args.training_protocol == 'legacy' and args.val_dataset == 'test':
        logger.warning('Legacy IRRA evaluates test during training; do not use this for SCFP model selection')

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader, args.eval_chunk_size) if val_img_loader is not None else None

    start_epoch = 1
    best_top1 = best_epoch = None
    if args.init_checkpoint:
        checkpointer.load(args.init_checkpoint, warm_start=True)
    if args.resume_checkpoint:
        checkpoint = checkpointer.resume(args.resume_checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_top1, best_epoch = checkpoint.get('best_top1'), checkpoint.get('best_epoch')

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler,
             checkpointer, best_top1, best_epoch)
