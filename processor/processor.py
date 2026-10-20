import logging
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from utils.checkpoint import unwrap
from experiments.scfp.diagnostics import (append_jsonl, write_json, gradient_norms,
                                          module_diagnostics, fixed_sample_diagnostics)


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, best_top1=None, best_epoch=None):
    device = next(model.parameters()).device
    logger = logging.getLogger('IRRA.train')
    master = get_rank() == 0
    core = unwrap(model)
    writer = SummaryWriter(args.output_dir) if master else None
    fixed_batch = None
    fixed_path = Path(args.output_dir) / 'fixed_diagnostics_batch.pt'
    if master:
        (Path(args.output_dir) / 'validation_metrics.jsonl').touch(exist_ok=True)
        if args.scfp_diagnostics and fixed_path.exists():
            fixed_batch = torch.load(fixed_path, map_location=device, weights_only=True)
    last_metrics = {}
    try:
        for epoch in range(start_epoch, args.num_epoch + 1):
            started = time.perf_counter()
            model.train()
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            meters = {}
            steps = 0
            for n_iter, batch in enumerate(train_loader):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                if master and args.scfp_diagnostics and fixed_batch is None:
                    fixed_batch = {k: v[:4].detach().clone() for k, v in batch.items()}
                    torch.save({k: v.cpu() for k, v in fixed_batch.items()}, fixed_path)
                ret = model(batch)
                total_loss = sum(v for k, v in ret.items() if k.endswith('_loss'))
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}, step {n_iter}')
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                log_now = n_iter == 0 or (n_iter + 1) % args.log_period == 0
                diagnostics = {}
                if master and args.scfp_diagnostics and log_now:
                    diagnostics = module_diagnostics(core, batch['pids'])
                    diagnostics['gradient_norms'] = gradient_norms(core)
                optimizer.step()
                for key, value in dict(ret, total_loss=total_loss).items():
                    meters.setdefault(key, AverageMeter()).update(float(value.detach()), len(batch['images']))
                steps += 1
                if master and log_now:
                    logger.info('Epoch %d step %d/%d: %s', epoch, steps, len(train_loader),
                                ', '.join(f'{k}={v.avg:.4f}' for k, v in meters.items() if 'loss' in k))
                    append_jsonl(Path(args.output_dir) / 'training_stats.jsonl',
                                 dict(kind='step', epoch=epoch, step=steps,
                                      losses={k: float(v.detach()) for k, v in ret.items() if k.endswith('_loss')},
                                      total_loss=float(total_loss.detach()), diagnostics=diagnostics))
                if args.max_steps and steps >= args.max_steps:
                    break
            if not steps:
                raise ValueError('Training loader is empty')
            scheduler.step()
            synchronize()
            if master:
                stats = dict(kind='epoch', epoch=epoch, steps=steps,
                             seconds=time.perf_counter() - started,
                             **{k: v.avg for k, v in meters.items()})
                append_jsonl(Path(args.output_dir) / 'training_stats.jsonl', stats)
                for key, meter in meters.items():
                    writer.add_scalar(key, meter.avg, epoch)
                writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
                if evaluator is not None and epoch % args.eval_period == 0:
                    top1 = evaluator.eval(core.eval(), full_metrics=False)
                    last_metrics = dict(epoch=epoch, split=args.val_dataset, metrics=evaluator.last_metrics)
                    append_jsonl(Path(args.output_dir) / 'validation_metrics.jsonl', last_metrics)
                    if best_top1 is None or top1 > best_top1:
                        best_top1, best_epoch = top1, epoch
                        checkpointer.save('best', epoch=epoch, best_top1=best_top1, best_epoch=best_epoch)
                if fixed_batch is not None:
                    fixed_sample_diagnostics(core, fixed_batch, args.output_dir, epoch)
                elif args.scfp_diagnostics:
                    write_json(Path(args.output_dir) / 'diagnostics.json', module_diagnostics(core))
                checkpointer.save('last', epoch=epoch, best_top1=best_top1, best_epoch=best_epoch)
                write_json(Path(args.output_dir) / 'metrics.json',
                           dict(protocol=args.training_protocol, validation=last_metrics,
                                best_top1=best_top1, best_epoch=best_epoch,
                                official_test='NOT RUN'))
                logger.info('Epoch %d saved; best validation R1=%s at epoch %s', epoch, best_top1, best_epoch)
            synchronize()
    finally:
        if writer:
            writer.close()


def do_inference(model, test_img_loader, test_txt_loader):
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    evaluator.eval(model.eval(), full_metrics=True)
    return evaluator.last_metrics
