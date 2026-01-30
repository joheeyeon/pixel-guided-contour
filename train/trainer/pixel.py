import torch.nn as nn
from .utils import (FocalLoss, DMLoss, sigmoid, TVLoss, CurvLoss, MDLoss, mIoULoss, EdgeStandardDeviationLoss, BoundedRegLoss,
                    SoftCELoss, FocalCELoss, CosineSimLoss, SoftBCELoss, MeanSimLoss, CDLoss)
import torch
import torch.nn.functional as F
import cv2, random
import numpy as np

class NetworkWrapper(nn.Module):
    def __init__(self, net, with_dml=True, ml_start_epoch=10, weight_dict=None, cfg=None):
        super(NetworkWrapper, self).__init__()
        self.cfg = cfg
        self.with_dml = with_dml
        self.ml_start_epoch = ml_start_epoch
        self.dml_start_epoch = self.cfg.train.dml_start_epoch
        self.mdml_start_epoch = self.cfg.train.mdml_start_epoch
        self.net = net
        self.weight_dict = weight_dict

        if 'pixel' in self.cfg.model.heads:
            if self.cfg.model.heads['pixel'] == 1:
                self.pix_crit = FocalLoss()
            else:
                pix_type = self.cfg.train.loss_type['pixel'] if 'pixel' in self.cfg.train.loss_type else 'ce'
                if pix_type == 'focal':
                    self.pix_crit = FocalCELoss(gamma=self.cfg.train.loss_params['pixel']['gamma'] if 'gamma' in self.cfg.train.loss_params['pixel'] else 2,
                                                reduce=self.cfg.train.loss_params['pixel']['reduce'] if 'reduce' in self.cfg.train.loss_params['pixel'] else True)
                else:
                    self.pix_crit = torch.nn.functional.cross_entropy

    def forward(self, batch, mode='default', output_t=None):
        if ('test' in batch['meta']) and (mode=='default'):
            output = self.net(batch['inp'], batch=batch)
            return output
        else:
            out_ontraining = {}
            output = self.net(batch['inp'], batch=batch)
            epoch = batch['epoch']
            scalar_stats = {}
            loss = 0.

            keyPointsMask = batch['keypoints_mask'][batch['ct_01']]
            # pixel
            if 'pixel' in self.cfg.model.heads:
                pixel_gt = F.interpolate(batch['pixel_gt'].unsqueeze(1).float(), size=(output['pixel'].size(-2), output['pixel'].size(-1)), mode='nearest').squeeze(1)
                if self.cfg.model.heads['pixel'] == 1:
                    pix_loss = self.pix_crit(sigmoid(output['pixel']), pixel_gt.bool().float())
                else:
                    pix_loss = self.pix_crit(output['pixel'], pixel_gt.bool().long())
                scalar_stats.update({'pix_loss': pix_loss})
                weight_pix = 1. if 'pixel' not in self.weight_dict else self.weight_dict['pixel']
                loss += weight_pix * pix_loss

            scalar_stats.update({'loss': loss})

            return output, loss, scalar_stats, out_ontraining

    def compute_loss(self, output, batch, output_t=None, mode='default'):
        """train_net_lit.py에서 사용하는 loss 계산 인터페이스"""
        out_ontraining = {}
        scalar_stats = {}
        loss = torch.tensor(0.0, device=output['pixel'].device, requires_grad=True)

        # pixel loss 계산
        if 'pixel' in self.cfg.model.heads:
            pixel_gt = F.interpolate(batch['pixel_gt'].unsqueeze(1).float(), 
                                     size=(output['pixel'].size(-2), output['pixel'].size(-1)), 
                                     mode='nearest').squeeze(1)
            if self.cfg.model.heads['pixel'] == 1:
                pix_loss = self.pix_crit(sigmoid(output['pixel']), pixel_gt.bool().float())
            else:
                pix_loss = self.pix_crit(output['pixel'], pixel_gt.bool().long())
            scalar_stats.update({'pix_loss': pix_loss})
            weight_pix = 1. if 'pixel' not in self.weight_dict else self.weight_dict['pixel']
            loss = loss + weight_pix * pix_loss

        scalar_stats.update({'loss': loss})

        return output, loss, scalar_stats, out_ontraining

    @torch.no_grad()
    def _create_targets(self, instances, img_hw, lid=0):
        # if self.predict_in_box_space:
        #     if self.clip_to_proposal or not self.use_rasterized_gt:  # in coco, this is true
        #         clip_boxes = torch.cat([inst.proposal_boxes.tensor for inst in instances])
        #         masks = self.clipper(instances, clip_boxes=clip_boxes, lid=lid)  # bitmask
        #     else:
        #         masks = rasterize_instances(self.gt_rasterizer, instances, self.rasterize_at)
        # else:
        #     masks = self.get_bitmasks(instances, img_h=img_wh[1], img_w=img_wh[0])
        masks = []
        for obj_i in range(instances.shape[0]):
            instance = instances[obj_i].astype(np.int32)
            mask = np.zeros((img_hw[0], img_hw[1], 1), np.uint8)
            masks.append(cv2.fillPoly(mask, [instance], 1))
        masks = np.stack(masks, axis=0)  # (Nc, H, W, 1)
        return masks.squeeze(-1)

