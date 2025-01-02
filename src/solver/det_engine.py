"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils

def to(module, device):
    return module.to(device) if hasattr(module, 'to') else module


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    criterion = to(criterion, device)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    ema = to(ema, device)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # TODO: check how dfine generates targets from image and bbox 
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]    # preprocess tarets
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        torch.use_deterministic_algorithms(True, warn_only=True)    # my modification. is this correct?
        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs, _ = model(samples, targets=targets)
                # out, dual_out

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            
            scaler.scale(loss).backward()
            
            if max_norm > 0:    # true
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # do I have to put the optimizer to device?
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:   # no scalar
            
            outputs, _ = model(samples, targets=targets)
            # out, dual_out
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()    # update model parameters

        if ema is not None:
            ema.update(model)   # update model parameters

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    torch.save(model.state_dict(), './my_save_ep=' + epoch + '.pt')
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, model, ema

from utils.general import xyxy2xywh, coco80_to_coco91_class
from pathlib import Path

def my_save_one_json(result, jdict, image_id, class_map):
    # predn: each pred [84, 300]    # 4 + 80    (originally [84, 5292])
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}

    # result = dict(labels=lab, boxes=box, scores=sco)
    # result of each sample contains:
        # box torch.Size([300, 4])  # xyxy
        # sco torch.Size([300])
        # lab torch.Size([300])

    boxes = xyxy2xywh(result['boxes'])  # in: n, 4 boxes, out: cxcywh
    boxes[:, :2] -= boxes[:, 2:] / 2  # xy center to top-left corner
    for box, score, lbl in zip(boxes.tolist(), result['scores'].tolist(), result['labels'].tolist()):
        jdict.append({
            'image_id': image_id.item(),
            'category_id': lbl,    # class map?
            'bbox': [round(coord, 3) for coord in box],
            'score': round(score, 5)})   # what's this? why is it always so low?

def my_save_json(results, jdict, targets, class_map):
    # results len 64
    # targets len 64
    for res, target in zip(results, targets):
        my_save_one_json(res, jdict, target['image_id'], class_map)


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    print('dfine eval')
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    # TODO: write prediction.json from postprocessed results!
    jdict = []
    class_map = coco80_to_coco91_class()

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # targets len 64
        # print('target of one sample', targets[0]['image_id'])   # tensor([13291], device='cuda:0')
        
        # samples are all ([128, 3, 640, 640])
        # model.to(device)
        outputs, _ = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes tensor([[640, 426]], as expected
        # orig_target_sizes = torch.tensor([[samples.shape[-1], samples.shape[-2]]], device=samples.device)

        results = postprocessor(outputs, orig_target_sizes)

        my_save_json(results, jdict, targets, class_map)
        # results len 64
        # a batch of results w each of them containing:
            # box torch.Size([300, 4])  # xyxy
            # sco torch.Size([300])
            # lab torch.Size([300])

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()

    # # accumulate predictions from all images
    # if coco_evaluator is not None:
    #     print('accumulate')
    #     coco_evaluator.accumulate()
    #     print('summarize')
    #     coco_evaluator.summarize()

    # stats = {}
    # # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if coco_evaluator is not None:
    #     if 'bbox' in iou_types:
    #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    #     if 'segm' in iou_types:
    #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return coco_evaluator, jdict
