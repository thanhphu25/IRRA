import logging
import time
import torch
import json
import os
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = next(model.parameters()).device
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = getattr(args, 'best_top1', -1.0)
    best_epoch = getattr(args, 'best_epoch', 0)
    if not len(train_loader):
        raise ValueError('Empty training loader')

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, iteration {n_iter}: {ret}')

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            for key in meters:
                if key != 'loss':
                    value = ret.get(key, 0)
                    meters[key].update(value.detach().item() if torch.is_tensor(value) else value,
                                       1 if key == 'mlm_acc' else batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
                target = model.module if args.distributed else model
                if getattr(args, 'lgsp_diagnostics', False):
                    logger.info('LGSP: %s', target.spatial_prompt.last_diagnostics)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
            performance = dict(epoch=epoch, seconds=end_time - start_time,
                seconds_per_batch=time_per_batch,
                peak_allocated_vram_gb=torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else None,
                losses={k: v.avg for k, v in meters.items() if 'loss' in k})
            target = model.module if args.distributed else model
            if getattr(args, 'lgsp_diagnostics', False):
                performance['lgsp'] = target.spatial_prompt.last_diagnostics
            with open(os.path.join(args.output_dir, 'training_stats.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(performance) + '\n')
        if epoch % eval_period == 0 or epoch == num_epoch:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval(), full_metrics=False)
                else:
                    top1 = evaluator.eval(model.eval(), full_metrics=False)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    best_epoch = epoch
                    arguments["epoch"] = epoch
                    arguments['best_top1'] = best_top1
                    arguments['best_epoch'] = best_epoch
                    checkpointer.save("best", **arguments)
                with open(os.path.join(args.output_dir, 'validation_metrics.jsonl'), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(epoch=epoch, split=args.val_dataset, metrics=evaluator.last_metrics)) + '\n')
        if get_rank() == 0:
            checkpointer.save('last', epoch=epoch, best_top1=best_top1, best_epoch=best_epoch)
        synchronize()
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {best_epoch}")
    tb_writer.close()


def do_inference(model, test_img_loader, test_txt_loader, output_file=None, chunk_size=128):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, chunk_size=chunk_size)
    top1 = evaluator.eval(model.eval())
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluator.last_metrics, f, indent=2)
    return evaluator.last_metrics
