"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ...core import register


__all__ = ['DFINEPostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class DFINEPostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        # outputs: a batch of raw predictions
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # logits torch.Size([64, 300, 80])  # there's batch info
        # boxes torch.Size([64, 300, 4])

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            logits_sigged = scores
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = {'labels': labels, 'boxes': boxes, 'scores': scores, 'logits': logits_sigged}
        return results
        # labels torch.Size([64, 300])
        # boxes torch.Size([64, 300, 4])
        # scores torch.Size([64, 300])
        
        results = []
        for lab, box, sco in zip(labels, boxes, scores):    # class_label, bbox, class_conf
            # box torch.Size([300, 4])  # xyxy
            # sco torch.Size([300])
            # lab torch.Size([300])
            # print('sco', sco)
            # print('lbl', lab)
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        return results  # a batch of results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
