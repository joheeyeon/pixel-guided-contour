import torch.nn as nn
from .utils import FocalLoss, sigmoid, DMLoss, TVLoss, CurvLoss, mIoULoss, PolyLoss, DouglasTorch, IndL1Loss1d, check_simply_connected, MDLoss, SelfConnectionPenalty
import torch


class NetworkWrapper(nn.Module):
    def __init__(self, net, with_dml=True, ml_start_epoch=10, weight_dict=None, cfg=None):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.with_dml = with_dml
        self.dml_start_epoch = ml_start_epoch
        self.weight_dict = weight_dict
        self.cfg = cfg

        self.ct_crit = FocalLoss()
        self.wh_crit = IndL1Loss1d('smooth_l1')
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.tv_crit = TVLoss(type=cfg.train.loss_type['tv'] if cfg is not None else 'smooth_l1')
        if self.cfg.train.with_iou_loss:
            self.region_crit = PolyLoss(cfg=cfg)
        if cfg.train.with_mdl:
            self.ml_crit = MDLoss(type=cfg.train.loss_type['md'] if cfg is not None else 'smooth_l1',
                                  match_with_ini=cfg.train.ml_match_with_ini if cfg is not None else True)
        elif with_dml:
            self.ml_crit = DMLoss(type='smooth_l1')
        else:
            self.ml_crit = self.py_crit
        if ('simple' in self.cfg.train.weight_dict) or ('simple_coarse' in self.cfg.train.weight_dict):
            self.simple_penalty = SelfConnectionPenalty(getattr(self.cfg.train,'loss_params')['simple'] if 'simple' in getattr(self.cfg.train,'loss_params') else {})

    def forward(self, batch, mode='default'):
        if 'test' in batch['meta']:
            output = self.net(batch['inp'], batch=batch)
            return output
        elif mode == 'preview':
            with torch.no_grad():
                output = self.net(batch['inp'], batch=batch)
                is_py_simple = check_simply_connected(output['py_pred'][0])
            return output, is_py_simple
        else:
            out_ontraining = {}

            output = self.net(batch['inp'], batch=batch)
            if ('simple' in self.cfg.train.save_ontraining) or (self.cfg.train.iou_params['schedule_type'] == 'after_simple'):
                with torch.no_grad():
                    is_py_simple = check_simply_connected(output['py_pred'][0])
                out_ontraining['is_simple'] = is_py_simple.clone().detach().cpu().numpy()
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

            wh_loss = self.wh_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01'])
            scalar_stats.update({'wh_loss': wh_loss})
            loss += self.weight_dict['box_wh'] * wh_loss

            # index- init & coarse
            if self.cfg.model.with_img_idx:
                poly_init = output['ex_pred'][batch['ct_01']]
                poly_coarse = output['py_pred'][0][batch['ct_01']]
            else:
                poly_init = output['ex_pred']
                poly_coarse = output['py_pred'][0]

            num_polys = len(output['ex_pred'])
            if num_polys == 0:
                init_py_loss = torch.sum(output['ex_pred']) * 0.
                coarse_py_loss = torch.sum(output['py_pred'][0]) * 0.
            else:
                init_py_loss = self.py_crit(output['ex_pred'], output['img_gt_init_polys'])
                coarse_py_loss = self.py_crit(output['py_pred'][0], output['img_gt_polys'])
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
                        out_ontraining['gt_coarse'] = output['img_gt_polys'].clone().detach().cpu().numpy()

            ## total variation
            if ('tv' in self.weight_dict) or ('tv_init' in self.weight_dict) or ('tv_coarse' in self.weight_dict):
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

                if 'tv_coarse' in self.weight_dict:
                    weight_tv_coarse = self.weight_dict['tv_coarse']
                else:
                    weight_tv_coarse = self.weight_dict['coarse'] * self.weight_dict['tv']

                if weight_tv_coarse > 0:
                    if num_polys == 0:
                        coarse_tv_loss = torch.sum(poly_coarse) * 0.
                    else:
                        coarse_tv_loss = self.tv_crit(poly_coarse)
                    scalar_stats.update({'coarse_tv_loss': coarse_tv_loss})
                    loss += coarse_tv_loss * weight_tv_coarse

            ## region
            mode_with_iou = False
            if mode == 'add_iou_loss':
                mode_with_iou = True
            elif self.cfg.train.iou_params['schedule_type'] == 'after_simple':
                mode_with_iou = False
            elif (self.cfg.train.with_iou_loss and (epoch >= self.cfg.train.start_epoch_region)):
                mode_with_iou = True
            if mode_with_iou:
                # FOR INIT
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
                            if 'py_init' not in out_ontraining:
                                out_ontraining['py_init'] = poly_init.clone().detach().cpu().numpy()
                                out_ontraining['gt_init'] = output['img_gt_init_polys'].clone().detach().cpu().numpy()
                            for key, val in out_region.items():
                                out_ontraining[f'region_init_{key}'] = val

                    #rev23: 24-09-09
                    # if region_loss_init < 0:
                    #     region_loss_init *= 0.
                    scalar_stats.update({f'region_init_loss': region_loss_init})
                    loss += region_loss_init * weight_region_crit_init
                # FOR COARSE
                if 'region_coarse' in self.weight_dict:
                    weight_region_crit_coarse = self.weight_dict['region_coarse']
                elif 'region' in self.weight_dict:
                    weight_region_crit_coarse = self.weight_dict['region'] * self.weight_dict['coarse']
                else:
                    weight_region_crit_coarse = self.weight_dict['coarse']

                if weight_region_crit_coarse > 0:
                    region_loss_coarse, out_region, is_py_simple = self.region_crit(poly_coarse, output['img_gt_polys'],
                                                        keypointsmask=keyPointsMask, save_mode=False if 'region' not in self.cfg.train.save_ontraining else self.cfg.train.save_ontraining['region'])
                    output['is_py_simple'] = {'coarse': is_py_simple}
                    if 'region' in self.cfg.train.save_ontraining:
                        if 'input' not in out_ontraining:
                            out_ontraining['input'] = batch['inp'].clone().detach().cpu().numpy()
                        if 'ct_01' not in out_ontraining:
                            out_ontraining['ct_01'] = batch['ct_01'].clone().detach().cpu().numpy()
                        if self.cfg.train.save_ontraining['region']:
                            if 'py_coarse' not in out_ontraining:
                                out_ontraining['py_coarse'] = poly_coarse.clone().detach().cpu().numpy()
                                out_ontraining['gt_coarse'] = output['img_gt_polys'].clone().detach().cpu().numpy()
                            for key, val in out_region.items():
                                out_ontraining[f'region_coarse_{key}'] = val

                    scalar_stats.update({f'region_coarse_loss': region_loss_coarse})
                    loss += region_loss_coarse * weight_region_crit_coarse

            # self-connection penalty
            if ('simple' in self.cfg.train.weight_dict) or ('simple_coarse' in self.cfg.train.weight_dict):
                if 'simple_coarse' in self.cfg.train.weight_dict:
                    weight_simple_coarse = self.cfg.train.weight_dict['simple_coarse']
                else:
                    weight_simple_coarse = self.cfg.train.weight_dict['simple'] * self.weight_dict['coarse']
                if weight_simple_coarse > 0:
                    simple_coars_loss = self.simple_penalty(poly_coarse)
                    scalar_stats.update({f'simple_coarse_loss': simple_coars_loss})
                    loss += simple_coars_loss * weight_simple_coarse
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

