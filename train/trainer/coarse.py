import torch.nn as nn
from .utils import FocalLoss, sigmoid, DMLoss, TVLoss, CurvLoss, mIoULoss, PolyLoss, DouglasTorch, check_simply_connected
import torch
import cv2, random
import numpy as np

class NetworkWrapper(nn.Module):
    def __init__(self, net, with_dml=True, ml_start_epoch=10, weight_dict=None, cfg=None):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.with_dml = with_dml
        self.dml_start_epoch = ml_start_epoch
        self.weight_dict = weight_dict
        self.cfg = cfg

        self.ct_crit = FocalLoss()
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.tv_crit = TVLoss(type=cfg.train.loss_type['tv'] if cfg is not None else 'smooth_l1')
        self.cv_crit = CurvLoss(type=cfg.train.loss_type['cv'] if cfg is not None else 'smooth_l1')
        if self.cfg.model.with_rasterize_net:
            self.region_crit = mIoULoss(n_classes=2)
        elif self.cfg.train.with_iou_loss:
            self.region_crit = PolyLoss(cfg=cfg)
        if with_dml:
            self.dml_crit = DMLoss(type=cfg.train.loss_type['dm'] if cfg is not None else 'smooth_l1')
        else:
            self.dml_crit = self.py_crit

        self.count_tmp = 0
        self.Douglas = DouglasTorch(
            num_vertices=cfg.train.iou_params['num_keypoints'] if 'num_keypoints' in cfg.train.iou_params else None,
            D=cfg.data.douglas['D'] if 'D' in cfg.data.douglas else 3,
            extract_type=cfg.data.douglas['extract_type'] if 'extract_type' in cfg.data.douglas else None)

    def forward(self, batch, mode='default'):
        if 'test' in batch['meta']:
            output = self.net(batch['inp'], batch=batch)
            return output
        elif mode == 'preview':
            with torch.no_grad():
                output = self.net(batch['inp'], batch=batch)
                is_py_simple = check_simply_connected(output['poly_coarse'])
            return output, is_py_simple
        else:
            out_ontraining = {}

            output = self.net(batch['inp'], batch=batch)
            epoch = batch['epoch']
            scalar_stats = {}
            loss = 0.

            if 'keypoints_mask' in batch:
                keyPointsMask = batch['keypoints_mask'][batch['ct_01']]
            else:
                keyPointsMask = None
            # for (backbone) ct & (train_decoder) init & coarse
            # ct
            ct_loss = self.ct_crit(sigmoid(output['ct_hm']), batch['ct_hm'])
            scalar_stats.update({'ct_loss': ct_loss})
            loss += self.weight_dict['box_ct'] * ct_loss
            # init & coarse
            if self.cfg.model.with_img_idx:
                poly_init = output['poly_init'][batch['ct_01']]
                poly_coarse = output['poly_coarse'][batch['ct_01']]
            else:
                poly_init = output['poly_init']
                poly_coarse = output['poly_coarse']

            num_polys = len(poly_init)
            if num_polys == 0:
                init_py_loss = torch.sum(poly_init) * 0.
                coarse_py_loss = torch.sum(poly_coarse) * 0.
            else:
                init_py_loss = self.py_crit(poly_init, output['img_gt_init_polys'])
                coarse_py_loss = self.py_crit(poly_coarse, output['img_gt_coarse_polys'])
            scalar_stats.update({'init_py_loss': init_py_loss})
            scalar_stats.update({'coarse_py_loss': coarse_py_loss})
            loss += init_py_loss * self.weight_dict['init']
            loss += coarse_py_loss * self.weight_dict['coarse']
            if 'py' in self.cfg.train.save_ontraining:
                if 'input' not in out_ontraining:
                    out_ontraining['input'] = batch['inp'].clone().detach().cpu().numpy()
                if 'ct_01' not in out_ontraining:
                    out_ontraining['ct_01'] = batch['ct_01'].clone().detach().cpu().numpy()
                if self.cfg.train.save_ontraining['py']:
                    if 'py_init' not in out_ontraining:
                        out_ontraining['py_init'] = poly_init.clone().detach().cpu().numpy()
                        out_ontraining['gt_init'] = output['img_gt_init_polys'].clone().detach().cpu().numpy()
                    if 'py_coarse' not in out_ontraining:
                        out_ontraining['py_coarse'] = poly_coarse.clone().detach().cpu().numpy()
                        out_ontraining['gt_coarse'] = output['img_gt_coarse_polys'].clone().detach().cpu().numpy()

            ## region
            mode_with_iou = False
            if mode == 'add_iou_loss':
                mode_with_iou = True
            elif self.cfg.train.iou_params['schedule_type'] == 'after_simple':
                mode_with_iou = False
            elif (self.cfg.train.with_iou_loss and (epoch >= self.cfg.train.start_epoch_region)):
                mode_with_iou = True
            if self.cfg.model.with_rasterize_net and (epoch >= self.cfg.train.start_epoch_region):
                from scipy.io import savemat #tmp-save
                import os
                save_dict = {} #tmp-save
                for py_name in ('init','coarse'):
                    gt_masks = self._create_targets(output[f'img_gt_{py_name}_polys'].clone().detach().cpu().numpy(),
                                                    [self.cfg.data.input_w, self.cfg.data.input_h])
                    if self.cfg.model.is_raster_down_sampled:
                        with torch.no_grad():
                            pred_masks = nn.functional.interpolate(output['pred_mask'][py_name], size=(self.cfg.data.input_w, self.cfg.data.input_h), mode='bilinear')
                    else:
                        pred_masks = output['pred_mask'][py_name]
                    region_loss = self.region_crit(pred_masks, torch.from_numpy(gt_masks).to(output['pred_mask'][py_name].device).long())
                    if 'region' in self.weight_dict:
                        weight_region_crit = self.weight_dict['region'] * self.weight_dict[py_name]
                    else:
                        weight_region_crit = self.weight_dict[py_name]

                    scalar_stats.update({f'region_{py_name}_loss': region_loss})
                    loss += region_loss * weight_region_crit

                    #tmp-save
                    pred_real_masks = self._create_targets(output[f'poly_{py_name}'].clone().detach().cpu().numpy(),
                                                    [self.cfg.data.input_w, self.cfg.data.input_h])
                    with torch.no_grad():
                        # pred_real_masks = nn.functional.interpolate(torch.from_numpy(pred_real_masks).to(output['pred_mask'][py_name].device).unsqueeze(1).float(),
                        #                                             size=(self.cfg.data.input_w, self.cfg.data.input_h), mode='bilinear')
                        pred_real_masks = torch.from_numpy(pred_real_masks).to(output['pred_mask'][py_name].device).unsqueeze(1)
                        pred_real_masks = torch.cat((1-pred_real_masks, pred_real_masks), dim=1)
                        real_region_loss = self.region_crit(pred_real_masks, torch.from_numpy(gt_masks).to(output['pred_mask'][py_name].device).long(), apply_softmax=False)
                    scalar_stats.update({f'real_region_{py_name}_loss': real_region_loss})
                    save_dict.update({f'{py_name}_pred_contour': output[f'poly_{py_name}'].clone().detach().cpu().numpy()})
                    save_dict.update({f"{py_name}_pred_real_masks": pred_real_masks})
                    save_dict.update({f'{py_name}_gt_masks': gt_masks})
                    save_dict.update({f'{py_name}_pred_masks': pred_masks.clone().detach().cpu().numpy()})
                # if epoch % 10 == 0:
                #     os.makedirs(f"{self.cfg.commen.result_dir}/OnTraining/raster/e{epoch}", exist_ok=True)
                #     savemat(f"{self.cfg.commen.result_dir}/OnTraining/raster/e{epoch}/{self.count_tmp}.mat",save_dict)
                self.count_tmp += 1

            elif mode_with_iou:
                if 'region_init' in self.weight_dict:
                    weight_region_crit_init = self.weight_dict['region_init']
                elif 'region' in self.weight_dict:
                    weight_region_crit_init = self.weight_dict['region'] * self.weight_dict['init']
                else:
                    weight_region_crit_init = self.weight_dict['init']

                if 'region_coarse' in self.weight_dict:
                    weight_region_crit_coarse = self.weight_dict['region_coarse']
                elif 'region' in self.weight_dict:
                    weight_region_crit_coarse = self.weight_dict['region'] * self.weight_dict['coarse']
                else:
                    weight_region_crit_coarse = self.weight_dict['coarse']

                if weight_region_crit_init > 0:
                    # region_loss_init = self.region_crit(poly_init, output['img_gt_init_polys'], keypointsmask=keyPointsMask,epoch=epoch)
                    region_loss_init, out_region, is_py_simple = self.region_crit(poly_init, output['img_gt_init_polys'],
                                                        keypointsmask=keyPointsMask, save_mode=False if 'region' not in self.cfg.train.save_ontraining else self.cfg.train.save_ontraining['region'])
                    output['is_py_simple'] = {'init': is_py_simple}
                    if 'region' in self.cfg.train.save_ontraining:
                        if 'input' not in out_ontraining:
                            out_ontraining['input'] = batch['inp'].clone().detach().cpu().numpy()
                        if 'ct_01' not in out_ontraining:
                            out_ontraining['ct_01'] = batch['ct_01'].clone().detach().cpu().numpy()
                        if self.cfg.train.save_ontraining['region']:
                            out_ontraining['py_init'] = poly_init.clone().detach().cpu().numpy()
                            out_ontraining['gt_init'] = output['img_gt_init_polys'].clone().detach().cpu().numpy()
                            for key, val in out_region.items():
                                out_ontraining[f'region_{key}'] = val
                    # if region_loss_init < 0:
                    #     region_loss_init *= torch.tensor(0.,device=region_loss_init.device)
                    scalar_stats.update({f'region_init_loss': region_loss_init})
                    loss += region_loss_init * weight_region_crit_init

                if weight_region_crit_coarse > 0:
                    # region_loss_coarse = self.region_crit(poly_coarse, output['img_gt_coarse_polys'], keypointsmask=keyPointsMask, epoch=epoch)
                    region_loss_coarse, out_region, is_py_simple = self.region_crit(poly_coarse, output['img_gt_coarse_polys'],
                                                          keypointsmask=keyPointsMask, save_mode=False if 'region' not in self.cfg.train.save_ontraining else self.cfg.train.save_ontraining['region'])
                    output['is_py_simple'] = {'coarse': is_py_simple}
                    if 'region' in self.cfg.train.save_ontraining:
                        if 'input' not in out_ontraining:
                            out_ontraining['input'] = batch['inp'].clone().detach().cpu().numpy()
                        if 'ct_01' not in out_ontraining:
                            out_ontraining['ct_01'] = batch['ct_01'].clone().detach().cpu().numpy()
                        if self.cfg.train.save_ontraining['region']:
                            out_ontraining['py_coarse'] = poly_coarse.clone().detach().cpu().numpy()
                            out_ontraining['gt_coarse'] = output['img_gt_coarse_polys'].clone().detach().cpu().numpy()
                            for key, val in out_region.items():
                                out_ontraining[f'region_{key}'] = val
                    # if region_loss_coarse < 0:
                    #     region_loss_coarse *= torch.tensor(0.,device=region_loss_coarse.device)
                    scalar_stats.update({f'region_coarse_loss': region_loss_coarse})
                    loss += region_loss_coarse * weight_region_crit_coarse

            ## total variation
            mode_with_tv = False
            if ('tv' in self.weight_dict) or ('tv_coarse' in self.weight_dict) or (
                    'tv_init' in self.weight_dict):
                if (self.cfg.train.iou_params['schedule_type'] == 'after_simple') and mode_with_iou and ('schedule_type_detail' not in self.cfg.train.iou_params):
                    mode_with_tv = False
                else:
                    mode_with_tv = True
            if mode_with_tv:
                if 'tv_init' in self.weight_dict:
                    weight_tv_init = self.weight_dict['tv_init']
                else:
                    weight_tv_init = self.weight_dict['init'] * self.weight_dict['tv']
                if 'tv_coarse' in self.weight_dict:
                    weight_tv_coarse = self.weight_dict['tv_coarse']
                else:
                    weight_tv_coarse = self.weight_dict['coarse'] * self.weight_dict['tv']

                if weight_tv_init > 0:
                    if num_polys == 0:
                        init_tv_loss = torch.sum(poly_init) * 0.
                    else:
                        init_tv_loss = self.tv_crit(poly_init)
                    scalar_stats.update({'init_tv_loss': init_tv_loss})
                    loss += init_tv_loss * weight_tv_init
                if weight_tv_coarse > 0:
                    if num_polys == 0:
                        coarse_tv_loss = torch.sum(poly_coarse) * 0.
                    else:
                        coarse_tv_loss = self.tv_crit(poly_coarse)

                    scalar_stats.update({'coarse_tv_loss': coarse_tv_loss})
                    loss += coarse_tv_loss * weight_tv_coarse

            if ('cv' in self.weight_dict) or ('cv_coarse' in self.weight_dict):
                if 'cv_init' in self.weight_dict:
                    weight_cv_init = self.weight_dict['cv_init']
                else:
                    weight_cv_init = self.weight_dict['init'] * self.weight_dict['cv']
                if 'cv_coarse' in self.weight_dict:
                    weight_cv_coarse = self.weight_dict['cv_coarse']
                else:
                    weight_cv_coarse = self.weight_dict['coarse'] * self.weight_dict['cv']

                if weight_cv_init > 0:
                    if num_polys == 0:
                        init_cv_loss = torch.sum(poly_init) * 0.
                    else:
                        init_cv_loss = self.cv_crit(poly_init)

                    scalar_stats.update({'init_cv_loss': init_cv_loss})
                    loss += init_cv_loss * weight_cv_init

                if weight_cv_coarse > 0:
                    if num_polys == 0:
                        coarse_cv_loss = torch.sum(poly_coarse) * 0.
                    else:
                        coarse_cv_loss = self.cv_crit(poly_coarse)

                    scalar_stats.update({'coarse_cv_loss': coarse_cv_loss})
                    loss += coarse_cv_loss * weight_cv_coarse

            if ('simple' in self.cfg.train.save_ontraining) or (
                    self.cfg.train.iou_params['schedule_type'] == 'after_simple'):
                if not mode_with_iou:
                    if 'schedule_type_detail' not in self.cfg.train.iou_params:
                        with torch.no_grad():
                            is_py_simple = check_simply_connected(output['poly_init'])
                        out_ontraining['is_simple'] = is_py_simple.clone().detach().cpu().numpy()
                    else:
                        with torch.no_grad():
                            is_simple = []
                            simple_preds = []
                            for ci in range(output['poly_coarse'].shape[0]):
                                simple_pred = output['poly_coarse'][ci, ...][
                                    self.Douglas.sample(output['poly_coarse'][ci, ...]).bool()]
                                is_simple.append(check_simply_connected(simple_pred).clone().detach().cpu().numpy())
                                simple_preds.append(simple_pred.clone().detach().cpu().numpy())
                        out_ontraining['is_simple'] = np.concatenate(is_simple, 0)
                        out_ontraining['simple_py'] = simple_preds

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