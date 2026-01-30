import torch.nn as nn
from .utils import FocalLoss, sigmoid, DMLoss, TVLoss, CurvLoss, mIoULoss, PolyLoss, DouglasTorch, check_simply_connected
import torch
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
        if self.cfg.train.with_iou_loss:
            self.region_crit = PolyLoss(cfg=cfg)
        if with_dml:
            self.dml_crit = DMLoss(type='smooth_l1')
        else:
            self.dml_crit = self.py_crit
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
                is_py_simple = check_simply_connected(output['poly_init'])
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
            # keyPointsMask = batch['keypoints_mask'][batch['ct_01']]
            # for (backbone) ct & (train_decoder) init & coarse
            # ct
            ct_loss = self.ct_crit(sigmoid(output['ct_hm']), batch['ct_hm'])
            scalar_stats.update({'ct_loss': ct_loss})
            loss += self.weight_dict['box_ct'] * ct_loss
            # init
            if self.cfg.model.with_img_idx:
                poly_init = output['poly_init'][batch['ct_01']]
            else:
                poly_init = output['poly_init']

            num_polys = len(output['poly_init'])
            if num_polys == 0:
                init_py_loss = torch.sum(output['poly_init']) * 0.
                # coarse_py_loss = torch.sum(output['poly_coarse']) * 0.
            else:
                init_py_loss = self.py_crit(output['poly_init'], output['img_gt_polys'])
                # coarse_py_loss = self.py_crit(output['poly_coarse'], output['img_gt_polys'])
            scalar_stats.update({'init_py_loss': init_py_loss})
            # scalar_stats.update({'coarse_py_loss': coarse_py_loss})
            loss += init_py_loss * self.weight_dict['init']
            # loss += coarse_py_loss * self.weight_dict['coarse']
            if 'py' in self.cfg.train.save_ontraining:
                if 'input' not in out_ontraining:
                    out_ontraining['input'] = batch['inp'].clone().detach().cpu().numpy()
                if 'ct_01' not in out_ontraining:
                    out_ontraining['ct_01'] = batch['ct_01'].clone().detach().cpu().numpy()
                if self.cfg.train.save_ontraining['py']:
                    if 'py_init' not in out_ontraining:
                        out_ontraining['py_init'] = poly_init.clone().detach().cpu().numpy()
                        out_ontraining['gt_init'] = output['img_gt_init_polys'].clone().detach().cpu().numpy()

            ## region
            mode_with_iou = False
            if mode == 'add_iou_loss':
                mode_with_iou = True
            elif self.cfg.train.iou_params['schedule_type'] == 'after_simple':
                mode_with_iou = False
            elif (self.cfg.train.with_iou_loss and (epoch >= self.cfg.train.start_epoch_region)):
                mode_with_iou = True
            if mode_with_iou:
                if 'region_init' in self.weight_dict:
                    weight_region_crit_init = self.weight_dict['region_init']
                elif 'region' in self.weight_dict:
                    weight_region_crit_init = self.weight_dict['region'] * self.weight_dict['init']
                else:
                    weight_region_crit_init = self.weight_dict['init']

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

                    #rev23: 24-09-09
                    # if region_loss_init < 0:
                    #     region_loss_init *= 0.
                    scalar_stats.update({f'region_init_loss': region_loss_init})
                    loss += region_loss_init * weight_region_crit_init

            ## total variation
            mode_with_tv = False
            if ('tv' in self.weight_dict) or ('tv_init' in self.weight_dict):
                if (self.cfg.train.iou_params['schedule_type'] == 'after_simple') and mode_with_iou and (
                        'schedule_type_detail' not in self.cfg.train.iou_params):
                    mode_with_tv = False
                else:
                    mode_with_tv = True
            if mode_with_tv:
                if 'tv_init' in self.weight_dict:
                    weight_tv_init = self.weight_dict['tv_init']
                else:
                    weight_tv_init = self.weight_dict['init'] * self.weight_dict['tv']

                if weight_tv_init > 0:
                    if num_polys == 0:
                        init_tv_loss = torch.sum(poly_init) * 0.
                    else:
                        init_tv_loss = self.tv_crit(poly_init)
                    scalar_stats.update({'init_tv_loss': init_tv_loss})
                    loss += init_tv_loss * weight_tv_init

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
                            for ci in range(output['poly_init'].shape[0]):
                                simple_pred = output['poly_init'][ci, ...][
                                    self.Douglas.sample(output['poly_init'][ci, ...]).bool()]
                                is_simple.append(check_simply_connected(simple_pred).clone().detach().cpu().numpy())
                                simple_preds.append(simple_pred.clone().detach().cpu().numpy())
                        out_ontraining['is_simple'] = np.concatenate(is_simple, 0)
                        out_ontraining['simple_py'] = simple_preds

            # for poly-lstm
            # py_loss = 0
            # n = len(output['py_pred']) - 1 if self.with_dml else len(output['py_pred'])
            # for i in range(n):
            #     if num_polys == 0:
            #         part_py_loss = torch.sum(output['py_pred'][i]) * 0.0
            #     else:
            #         part_py_loss = self.py_crit(output['py_pred'][i], output['img_gt_polys'])
            #     py_loss += part_py_loss / len(output['py_pred'])
            #     scalar_stats.update({'py_loss{}'.format(i): part_py_loss})
            # loss += py_loss * self.weight_dict['evolve']
            #
            # if self.with_dml:
            #     if epoch >= self.dml_start_epoch:
            #         dm_loss = self.dml_crit(output['py_pred'][-2],
            #                                 output['py_pred'][-1],
            #                                 output['img_gt_polys'],
            #                                 keyPointsMask)
            #         scalar_stats.update({'py_final_loss': dm_loss})
            #         loss += dm_loss / len(output['py_pred']) * self.weight_dict['evolve']
            #     else:
            #         py_last_loss = self.py_crit(output['py_pred'][-1], output['img_gt_polys'])
            #         scalar_stats.update({'py_final_loss': py_last_loss})
            #         loss += py_last_loss / len(output['py_pred']) * self.weight_dict['evolve']

            scalar_stats.update({'loss': loss})

            return output, loss, scalar_stats, out_ontraining

