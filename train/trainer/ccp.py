"""
Snake Energy Loss Terms in CCP Framework:

1. Elastic Term (Continuity): TVLoss
   - TV (Total Variation) loss: 연속성을 강제하여 contour가 부드럽게 연결되도록
   - Snake energy의 α * |ds/dx|² term에 대응
   - weight key: 'tv', 'tv_init', 'tv_coarse', 'tv_evolve', 'tv_evolve_{i}'

2. Bending Term (Curvature): CurvLoss
   - Snake energy의 β * |d²s/dx²|² term에 대응
   - weight key: 'cv', 'cv_init', 'cv_coarse', 'cv_evolve', 'cv_evolve_{i}'
   - loss_type 'l2' 설정시 정확한 snake energy와 일치

3. External Energy: pixel loss, ct loss 등
   - 이미지 feature에 기반한 external force
   - Snake energy의 external energy term에 대응

Configuration 예시:
weight_dict = {
    'tv': 1.0,      # elastic term weight
    'cv': 0.1,      # bending term weight  
}

loss_type = {
    'tv': 'l2',     # elastic loss type
    'cv': 'l2',     # bending loss type (정확한 snake energy)
}
"""

import torch.nn as nn
from .utils import (FocalLoss, DMLoss, sigmoid, TVLoss, CurvLoss, MDLoss, mIoULoss, EdgeStandardDeviationLoss, BoundedRegLoss,
                    SoftCELoss, FocalCELoss, CosineSimLoss, SoftBCELoss, MeanSimLoss, CDLoss, VertexClsLoss, FocalBCELoss, TemperatureFocalCELoss)
import torch
import torch.nn.functional as F
import cv2, random
import numpy as np
import torch.distributed as dist

def check_nan(name, tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[NaN detected] {name} has NaN or Inf!")

def safe_empty_tensor(tensor):
    if tensor.numel() == 0:
        # tensor shape 예: (0, 64, 2) => (1, 64, 2) 이렇게 수정
        shape = list(tensor.shape)
        if shape[0] == 0:
            shape[0] = 1  # batch dim만 1로 변경
        empty_tensor = torch.zeros(
            shape,
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=tensor.requires_grad
        )
        return empty_tensor
    else:
        return tensor

def safe_output(output):
    def _safe(v):
        if isinstance(v, torch.Tensor):
            return safe_empty_tensor(v)
        elif isinstance(v, list):
            return [_safe(t) for t in v]
        elif isinstance(v, dict):
            return {kk: _safe(vv) for kk, vv in v.items()}
        else:
            return v

    return {k: _safe(v) for k, v in output.items()}


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
        self.loss_dict = nn.ModuleDict()

        if cfg.train.weight_dict.get("vertex_cls", 0.) > 0:
            self.loss_dict["vertex_cls"] = VertexClsLoss(**self.cfg.train.loss_params["vertex_cls"])

        if 'pixel' in self.cfg.model.heads:
            if self.cfg.model.heads['pixel'] == 1:
                self.pix_crit = FocalLoss()
                self.pix_type = 'focal_single'
            else:
                self.pix_type = self.cfg.train.loss_type['pixel'] if 'pixel' in self.cfg.train.loss_type else 'ce'
                if self.pix_type == 'focal':
                    self.pix_crit = FocalCELoss(gamma=self.cfg.train.loss_params['pixel']['gamma'] if 'gamma' in self.cfg.train.loss_params['pixel'] else 2,
                                                reduce=self.cfg.train.loss_params['pixel']['reduce'] if 'reduce' in self.cfg.train.loss_params['pixel'] else True)
                elif self.pix_type == 'focal_bce':
                    gamma = self.cfg.train.loss_params['pixel']['gamma'] if 'gamma' in self.cfg.train.loss_params['pixel'] else 2.0
                    alpha_fg = self.cfg.train.loss_params['pixel']['alpha_fg'] if 'alpha_fg' in self.cfg.train.loss_params['pixel'] else 0.5
                    alpha_bg = self.cfg.train.loss_params['pixel']['alpha_bg'] if 'alpha_bg' in self.cfg.train.loss_params['pixel'] else 0.25
                    reduction = "mean" if self.cfg.train.loss_params['pixel'].get('reduce', True) else "none"
                    self.pix_crit = FocalBCELoss(gamma=gamma, alpha_fg=alpha_fg, alpha_bg=alpha_bg, reduction=reduction)
                elif self.pix_type == 'temperature_focal':
                    # trainable_softmax 타입일 때 사용되는 temperature focal CE loss
                    self.pix_crit = TemperatureFocalCELoss(gamma=self.cfg.train.loss_params['pixel']['gamma'] if 'gamma' in self.cfg.train.loss_params['pixel'] else 2,
                                                          reduce=self.cfg.train.loss_params['pixel']['reduce'] if 'reduce' in self.cfg.train.loss_params['pixel'] else True)
                else:
                    self.pix_crit = torch.nn.functional.cross_entropy

        self.ct_crit = FocalLoss()
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.tv_crit = TVLoss(type=cfg.train.loss_type['tv'] if cfg is not None else 'smooth_l1')
        self.cv_crit = CurvLoss(type=cfg.train.loss_type['cv'] if cfg is not None else 'smooth_l1')
        if self.cfg.model.with_rasterize_net:
            self.region_crit = mIoULoss(n_classes=2)

        if with_dml:
            self.ml_crit = DMLoss(type=cfg.train.loss_type['dm'] if cfg is not None else 'smooth_l1')
        elif cfg.train.with_mdl:
            self.ml_crit = MDLoss(type=cfg.train.loss_type['md'] if cfg is not None else 'smooth_l1',
                                  match_with_ini=cfg.train.ml_match_with_ini if cfg is not None else True)
        else:
            self.ml_crit = self.py_crit

        PY_RANGE_DICT = {'none': [],
                         'final': [self.cfg.model.evolve_iters - 1],
                         'penultimate': [self.cfg.model.evolve_iters - 2],
                         'last2': [i for i in range(self.cfg.model.evolve_iters - 2, self.cfg.model.evolve_iters)],
                         'except_1st': [i for i in range(1, self.cfg.model.evolve_iters)],
                         'all': [i for i in range(self.cfg.model.evolve_iters)]}
        self.ml_range_py = PY_RANGE_DICT[self.cfg.train.ml_range_py]
        if self.cfg.train.dml_range != 'none':
            self.dml_range = PY_RANGE_DICT[self.cfg.train.dml_range]
            self.dml_crit = DMLoss(type=cfg.train.loss_type['dm'] if cfg is not None else 'smooth_l1')
        else:
            self.dml_range = []
        if self.cfg.train.mdml_range != 'none':
            self.mdml_range = PY_RANGE_DICT[self.cfg.train.mdml_range]
            self.mdml_crit = MDLoss(type=cfg.train.loss_type['md'] if cfg is not None else 'smooth_l1',
                                  match_with_ini=cfg.train.ml_match_with_ini if cfg is not None else True)
        else:
            self.mdml_range = []
        if ('edge_std' in self.cfg.train.weight_dict) or ('edge_std_init' in self.cfg.train.weight_dict) or ('edge_std_coarse' in self.cfg.train.weight_dict) or ('edge_std_evolve' in self.cfg.train.weight_dict):
            self.eeq_crit = EdgeStandardDeviationLoss()
        else:
            self.eeq_crit = None
        if self.cfg.model.with_sharp_contour:
            self.ipc_crit = nn.BCEWithLogitsLoss()
        if 'kd' in cfg.commen.task:
            self.kd_cls_crit = SoftCELoss(T=self.cfg.train.kd_param['soft_T'] if 'soft_T' in self.cfg.train.kd_param else 10.)
            self.kd_bcls_crit = SoftBCELoss(
                T=self.cfg.train.kd_param['soft_T'] if 'soft_T' in self.cfg.train.kd_param else 10.)
            self.kd_reg_crit = BoundedRegLoss(type=cfg.train.loss_type['kd_reg'] if 'kd_reg' in cfg.train.loss_type else 'smooth_l1',
                                              condition_type=cfg.train.kd_param['reg_condition_type'] if 'reg_condition_type' in cfg.train.kd_param else 'error',
                                              margin=cfg.train.kd_param['reg_margin'] if 'reg_margin' in cfg.train.kd_param else 0)
            inter_type = cfg.train.kd_param['inter_type'] if 'inter_type' in cfg.train.kd_param else 'cosine'
            if 'mean' in inter_type:
                self.kd_inter_crit = MeanSimLoss(sim_type=inter_type.split('_')[-1])
            elif 'cd' in inter_type:
                self.kd_inter_crit = CDLoss(soft_param=self.cfg.train.kd_param['feature_soft_param'] if 'feature_soft_param' in self.cfg.train.kd_param else 1)
            else:
                self.kd_inter_crit = CosineSimLoss(apply_type=self.cfg.train.kd_param[
                    'feature_apply_type'] if 'feature_apply_type' in self.cfg.train.kd_param else 'channel')

        # DDP-safe anchor param (가벼움; 모든 step에서 그래프 연결 보장), edit:debug:ddp-stop:25-08-09
        self.register_buffer("_ddp_anchor", torch.zeros((), dtype=torch.float32), persistent=False)

    def forward(self, batch, mode='default', output_t=None):
        # print(f"[NETWORK WRAPPER] meta={batch.get('meta')}")
        if 'test' in batch['meta']:
            output = self.net(batch['inp'], batch=batch)
            return output
        else:
            output = self.net(batch['inp'], batch=batch)
            return self.compute_loss(output, batch, output_t=output_t, mode=mode)

    def compute_loss(self, output, batch, output_t=None, mode='default'):
        out_ontraining = {}
        epoch = batch['epoch']
        scalar_stats = {}
        # 항상 텐서+그래프로 시작 (rank/step 동일하게 존재), DDP-safe, edit:debug:ddp-stop:25-08-09
        loss = self._ddp_anchor * 0.0
        dummy = self._ddp_anchor * 0.0

        keyPointsMask = batch['keypoints_mask'][batch['ct_01']]
        # vertex cls
        if self.cfg.train.weight_dict.get("vertex_cls", 0.) > 0:
            # 🚨 DDP-safe check: 'py_valid_logits'가 비어있는 경우를 처리합니다.
            if 'py_valid_logits' in output and len(output['py_valid_logits']) > 0:
                vertex_cls_loss = dummy
                for py_valid_logit in output['py_valid_logits']:
                    pred_vertex_logits = py_valid_logit.permute(0, 2, 1)
                    # print(pred_vertex_logits.shape)
                    pred_coords = output['py_pred'][-1]
                    vertex_gt_coord = output['img_gt_polys']
                    vertex_cls_loss += self.loss_dict["vertex_cls"](pred_vertex_logits, pred_coords, vertex_gt_coord, keyPointsMask)/len(output['py_valid_logits'])
            else:
                # 'py_valid_logits'가 없거나 비어있으면 dummy_loss를 사용하여 연산 그래프를 유지합니다.
                vertex_cls_loss = dummy
            scalar_stats.update({'vtx_cls_loss': vertex_cls_loss})
            weight_vertex_cls = self.cfg.train.weight_dict.get("vertex_cls", 0.)
            loss += weight_vertex_cls * vertex_cls_loss

        # pixel
        if 'pixel' in self.cfg.model.heads:
            if isinstance(output['pixel'], list):
                to_size = (output['pixel'][-1].size(-2), output['pixel'][-1].size(-1))
            else:
                to_size = (output['pixel'].size(-2), output['pixel'].size(-1))

            pixel_gt = F.interpolate(batch['pixel_gt'].unsqueeze(1).float(), size=to_size, mode='nearest').squeeze(1)
            for pixelmap_i in range(len(output['pixel'])):
                if self.cfg.model.heads['pixel'] == 1:
                    pix_loss = self.pix_crit(sigmoid(output['pixel'][pixelmap_i]), pixel_gt.bool().float())
                elif self.pix_type == 'focal_bce':
                    # focal_bce는 2 채널로 출력하고 binary target 사용
                    pix_loss = self.pix_crit(output['pixel'][pixelmap_i], pixel_gt.bool().long())
                elif self.pix_type == 'temperature_focal' and self.cfg.model.ccp_deform_pixel_norm in ['trainable_softmax', 'trainable_softmax_softclamp']:
                    # trainable_softmax 또는 trainable_softmax_softclamp 타입일 때 temperature 파라미터 사용
                    # Evolution 모듈에서 temperature 가져오기 - gcn이 Evolution 모듈이므로 직접 접근
                    temperature = None
                    
                    # trainable_softmax: self.net.gcn.temperature 사용
                    if hasattr(self.net, 'gcn') and hasattr(self.net.gcn, 'temperature'):
                        temperature = self.net.gcn.temperature
                    # trainable_softmax_softclamp: u 파라미터에서 temperature 계산
                    elif hasattr(self.net, 'gcn') and hasattr(self.net.gcn, 'u') and hasattr(self.net.gcn, 'T_lo') and hasattr(self.net.gcn, 'T_hi'):
                        # T = T_lo + (T_hi - T_lo) * sigmoid(u)
                        u = self.net.gcn.u
                        T_lo = self.net.gcn.T_lo
                        T_hi = self.net.gcn.T_hi
                        temperature = T_lo + (T_hi - T_lo) * torch.sigmoid(u)
                    
                    if temperature is not None:
                        pix_loss = self.pix_crit(output['pixel'][pixelmap_i], pixel_gt.bool().long(), temperature)
                    else:
                        # fallback to regular focal CE loss (use FocalCELoss instead of raw cross_entropy)
                        fallback_focal = FocalCELoss(gamma=2, reduce=True)
                        pix_loss = fallback_focal(output['pixel'][pixelmap_i], pixel_gt.bool().long())
                else:
                    if hasattr(self, 'pix_crit') and callable(self.pix_crit):
                        pix_loss = self.pix_crit(output['pixel'][pixelmap_i], pixel_gt.bool().long())
                    else:
                        # deterministic safe fallback using FocalCELoss
                        fallback_focal = FocalCELoss(gamma=2, reduce=True)
                        pix_loss = fallback_focal(output['pixel'][pixelmap_i], pixel_gt.bool().long())
                scalar_stats.update({f'pix_loss{pixelmap_i}': pix_loss})
                if f'pixel_{pixelmap_i}' in self.weight_dict:
                    weight_pix = self.weight_dict[f'pixel_{pixelmap_i}']
                elif 'pixel' in self.weight_dict:
                    weight_pix = self.weight_dict['pixel']
                else:
                    weight_pix = 1.
                if self.cfg.train.is_normalize_pixel:
                    loss += weight_pix * pix_loss/len(output['pixel'])
                else:
                    loss += weight_pix * pix_loss

        # ct
        if (output_t is not None) and ('ct' in self.cfg.train.kd_param['losses']) and (
                self.cfg.train.kd_param['weight_type'] == 'normalized'):
            if f'kd_ct' in self.weight_dict:
                weight_kd = self.weight_dict[f'kd_ct']
            elif 'kd' in self.weight_dict:
                weight_kd = self.weight_dict['kd']
            else:
                weight_kd = 0.5
            if weight_kd >= 1.:
                weight_kd = weight_kd / (1 + weight_kd)
            weight_ct = self.weight_dict['box_ct'] * (1 - weight_kd)
        else:
            weight_ct = self.weight_dict['box_ct']

        if self.weight_dict['box_ct'] > 0:
            # 🚨 DDP-unsafe한 조건문 수정: ct_hm이 0인 경우에도 dummy_loss를 사용하여 연산 그래프를 유지합니다.
            if batch['ct_hm'].sum() == 0 and epoch < 5:
                ct_loss = dummy
            else:
                ct_loss = self.ct_crit(sigmoid(output['ct_hm']), batch['ct_hm'])
            # print(f"({dist.get_rank()})[DEBUG] ct_loss requires_grad:", ct_loss.requires_grad)
            scalar_stats.update({'ct_loss': ct_loss})
            loss += weight_ct * ct_loss

        # init & coarse
        if self.cfg.model.with_img_idx:
            poly_init = output['poly_init'][batch['ct_01']]
            poly_coarse = output['poly_coarse'][batch['ct_01']]
            py_pred = []
            # ccp task에서는 py_pred 대신 py를 사용
            if 'py_pred' in output:
                for py in output['py_pred']:
                    py_pred.append(py[batch['ct_01']])
            elif 'py' in output:
                for py in output['py']:
                    py_pred.append(py[batch['ct_01']])
            # Stage 1에서 evolve_iters=0이면 py_pred/py가 없을 수 있음 (빈 리스트로 유지)
        else:
            poly_init = output['poly_init']
            poly_coarse = output['poly_coarse']
            # ccp task에서는 py_pred 대신 py를 사용
            if 'py_pred' in output:
                py_pred = output['py_pred']
            elif 'py' in output:
                py_pred = output['py']
            else:
                # Stage 1에서 evolve_iters=0이면 py_pred가 없을 수 있음
                py_pred = []

        num_polys = len(poly_init)

        if num_polys == 0:
            init_py_loss = dummy
            coarse_py_loss = dummy
        else:
            # GT polygons: ccp_maskinit는 별도 키 사용, ccp는 img_gt_polys 공용
            gt_init_polys = output.get('img_gt_init_polys', output.get('img_gt_polys', []))
            gt_coarse_polys = output.get('img_gt_coarse_polys', output.get('img_gt_polys', []))
            
            # GT가 없으면 dummy loss 사용
            if len(gt_init_polys) == 0 or len(gt_coarse_polys) == 0:
                init_py_loss = dummy
                coarse_py_loss = dummy
            else:
                # print(f"poly_init] :  {poly_init.max()}, gt_init_polys : {gt_init_polys.max()}")
                init_py_loss = self.py_crit(poly_init, gt_init_polys)
                coarse_py_loss = self.py_crit(poly_coarse, gt_coarse_polys)

        # print(f"({dist.get_rank()})[DEBUG] init_py_loss requires_grad:", init_py_loss.requires_grad)
        # print(f"({dist.get_rank()})[DEBUG] coarse_py_loss requires_grad:", coarse_py_loss.requires_grad)
        if (output_t is not None) and ('init' in self.cfg.train.kd_param['losses']) and (
                self.cfg.train.kd_param['weight_type'] == 'normalized'):
            if f'kd_init' in self.weight_dict:
                weight_kd = self.weight_dict[f'kd_init']
            elif 'kd' in self.weight_dict:
                weight_kd = self.weight_dict['kd']
            else:
                weight_kd = 0.5
            if weight_kd >= 1.:
                weight_kd = weight_kd / (1 + weight_kd)
            weight_py = self.weight_dict['init'] * (1 - weight_kd)
            print(f"weight_py (with kd) : {weight_py}")
        else:
            weight_py = self.weight_dict['init']
        
        
        if self.weight_dict['init'] > 0 and weight_py > 0:
            scalar_stats.update({'init_py_loss': init_py_loss})
            loss += init_py_loss * weight_py

        if (output_t is not None) and ('coarse' in self.cfg.train.kd_param['losses']) and (
                self.cfg.train.kd_param['weight_type'] == 'normalized'):
            if f'kd_coarse' in self.weight_dict:
                weight_kd = self.weight_dict[f'kd_coarse']
            elif 'kd' in self.weight_dict:
                weight_kd = self.weight_dict['kd']
            else:
                weight_kd = 0.5
            if weight_kd >= 1.:
                weight_kd = weight_kd / (1 + weight_kd)
            weight_py = self.weight_dict['coarse'] * (1 - weight_kd)
            print(f"weight_py (with kd) : {weight_py}")
        else:
            weight_py = self.weight_dict['coarse']
        if self.weight_dict['coarse'] > 0:
            scalar_stats.update({'coarse_py_loss': coarse_py_loss})
            loss += coarse_py_loss * weight_py

        if self.ml_range_py:
            special_loss_range = self.ml_range_py
            special_loss_start_epoch = self.ml_start_epoch
        elif self.dml_range:
            special_loss_range = self.dml_range
            special_loss_start_epoch = self.dml_start_epoch
        elif self.mdml_range:
            special_loss_range = self.mdml_range
            special_loss_start_epoch = self.mdml_start_epoch
        else:
            special_loss_range = []
            special_loss_start_epoch = 0

        # evolve loss는 py_pred가 있을 때만 계산 (Stage 1에서는 evolve_iters=0이므로 py_pred가 없을 수 있음)
        if self.weight_dict['evolve'] > 0 and len(py_pred) > 0:
            # py_loss = dummy
            n = len(py_pred)
            for i in range(n):
                if (output_t is not None) and (f'evolve_{i}' in self.cfg.train.kd_param['losses']) and (
                        self.cfg.train.kd_param['weight_type'] == 'normalized'):
                    if f'kd_py_{i}' in self.weight_dict:
                        weight_kd = self.weight_dict[f'kd_py_{i}']
                    elif 'kd' in self.weight_dict:
                        weight_kd = self.weight_dict['kd']
                    else:
                        weight_kd = 0.5
                    if weight_kd >= 1.:
                        weight_kd = weight_kd / (1 + weight_kd)
                    weight_py = self.weight_dict['evolve'] * (1 - weight_kd)
                    print(f"weight_py (with kd) : {weight_py}")
                else:
                    weight_py = self.weight_dict['evolve']
                # print(f"py_pred[i] :  {py_pred[i].max()}, output['img_gt_polys'] : {output['img_gt_polys'].max()}")
                if num_polys == 0:
                    part_loss = dummy
                else:
                    # 🚨 DDP-safe check: Ground Truth가 없는 경우를 처리합니다.
                    if i in special_loss_range and epoch >= special_loss_start_epoch:
                        if len(output['img_gt_polys']) == 0:
                            part_loss = dummy
                        else:
                            if self.with_dml:
                                part_loss = self.ml_crit(py_pred[i - 1], py_pred[i], output['img_gt_polys'],
                                                         keyPointsMask)
                            elif self.cfg.train.with_mdl:
                                part_loss = self.ml_crit(py_pred[i - 1], py_pred[i], output['img_gt_polys']).mean()
                            else:
                                part_loss = self.py_crit(py_pred[i], output['img_gt_polys'])
                    else:
                        part_loss = self.py_crit(py_pred[i], output['img_gt_polys'])

                scalar_stats.update({f'py_loss_{i}': part_loss})
                loss += part_loss / len(py_pred) * weight_py

        ## total variation
        tv_weight_keys = [k for k in self.weight_dict.keys() if k.startswith('tv')]
        if tv_weight_keys:
            if num_polys == 0:
                init_tv_loss = dummy
                coarse_tv_loss = dummy
            else:
                init_tv_loss = self.tv_crit(poly_init)
                coarse_tv_loss = self.tv_crit(poly_coarse)
            if 'tv_init' in self.weight_dict:
                weight_tv_init = self.weight_dict['tv_init']
            else:
                weight_tv_init = self.weight_dict.get('init', 0) * self.weight_dict.get('tv', 0)
            if 'tv_coarse' in self.weight_dict:
                weight_tv_coarse = self.weight_dict['tv_coarse']
            else:
                weight_tv_coarse = self.weight_dict['coarse'] * self.weight_dict['tv']
            # tv_evolve weight 처리: 각 iteration별로 다른 weight 적용 가능
            # tv_evolve_0, tv_evolve_1, ... 형태로 개별 weight 설정 가능
            # 없으면 tv_evolve 사용, 그것도 없으면 evolve * tv 사용
            if weight_tv_init > 0:
                scalar_stats.update({'init_tv_loss': init_tv_loss})
                loss += init_tv_loss * weight_tv_init
            if weight_tv_coarse > 0:
                scalar_stats.update({'coarse_tv_loss': coarse_tv_loss})
                loss += coarse_tv_loss * weight_tv_coarse
            
            # evolve TV loss - iteration별 개별 weight 지원
            if len(py_pred) > 0 and 'tv' in self.weight_dict:
                for i in range(len(py_pred)):
                    # iteration별 개별 weight 확인
                    if f'tv_evolve_{i}' in self.weight_dict:
                        weight_tv_evolve_i = self.weight_dict[f'tv_evolve_{i}']
                    elif 'tv_evolve' in self.weight_dict:
                        weight_tv_evolve_i = self.weight_dict['tv_evolve']
                    else:
                        weight_tv_evolve_i = self.weight_dict['evolve'] * self.weight_dict['tv']
                    
                    if weight_tv_evolve_i > 0:
                        tv_loss_i = self.tv_crit(py_pred[i])
                        scalar_stats.update({f'evolve_tv_loss_{i}': tv_loss_i})
                        loss += tv_loss_i * weight_tv_evolve_i

        # CV loss 조건 수정: cv로 시작하는 모든 키를 확인
        cv_weight_keys = [k for k in self.weight_dict.keys() if k.startswith('cv')]
        if cv_weight_keys:
            if num_polys == 0:
                init_cv_loss = dummy
                coarse_cv_loss = dummy
            else:
                init_cv_loss = self.cv_crit(poly_init)
                coarse_cv_loss = self.cv_crit(poly_coarse)
                
            # Weight 계산: 개별 weight 우선, 없으면 stage * cv weight 사용
            if 'cv_init' in self.weight_dict:
                weight_cv_init = self.weight_dict['cv_init']
            else:
                weight_cv_init = self.weight_dict.get('init', 0) * self.weight_dict.get('cv', 0)
            if 'cv_coarse' in self.weight_dict:
                weight_cv_coarse = self.weight_dict['cv_coarse']
            else:
                weight_cv_coarse = self.weight_dict.get('coarse', 0) * self.weight_dict.get('cv', 0)
            
            if weight_cv_init > 0:
                scalar_stats.update({'init_cv_loss': init_cv_loss})
                loss += init_cv_loss * weight_cv_init
            if weight_cv_coarse > 0:
                scalar_stats.update({'coarse_cv_loss': coarse_cv_loss})
                loss += coarse_cv_loss * weight_cv_coarse
            
            # evolve CV loss - iteration별 개별 weight 지원 (TV loss와 동일하게)
            if len(py_pred) > 0:
                for i in range(len(py_pred)):
                    # iteration별 개별 weight 확인
                    if f'cv_evolve_{i}' in self.weight_dict:
                        weight_cv_evolve_i = self.weight_dict[f'cv_evolve_{i}']
                    elif 'cv_evolve' in self.weight_dict:
                        weight_cv_evolve_i = self.weight_dict['cv_evolve']
                    else:
                        weight_cv_evolve_i = self.weight_dict.get('evolve', 0) * self.weight_dict.get('cv', 0)
                    
                    if weight_cv_evolve_i > 0:
                        cv_loss_i = self.cv_crit(py_pred[i])
                        scalar_stats.update({f'evolve_cv_loss_{i}': cv_loss_i})
                        loss += cv_loss_i * weight_cv_evolve_i

        ## edge standard deviation loss (Edge Equal loss = eeq loss)
        if self.eeq_crit is not None:
            if num_polys == 0:
                init_eeq_loss = dummy
                coarse_eeq_loss = dummy
                evolve_eeq_loss = dummy
            else:
                init_eeq_loss = self.eeq_crit(poly_init)
                coarse_eeq_loss = self.eeq_crit(poly_coarse)
                evolve_eeq_loss = dummy
                for i in range(len(py_pred)):
                    evolve_eeq_loss += self.eeq_crit(py_pred[i]) / len(py_pred)

            if 'edge_std_init' in self.weight_dict:
                weight_eeq_init = self.weight_dict['edge_std_init']
            else:
                weight_eeq_init = self.weight_dict['init'] * self.weight_dict['edge_std']
            if 'edge_std_coarse' in self.weight_dict:
                weight_eeq_coarse = self.weight_dict['edge_std_coarse']
            else:
                weight_eeq_coarse = self.weight_dict['coarse'] * self.weight_dict['edge_std']
            if 'edge_std_evolve' in self.weight_dict:
                weight_eeq_evolve = self.weight_dict['edge_std_evolve']
            else:
                weight_eeq_evolve = self.weight_dict['evolve'] * self.weight_dict['edge_std']

            if weight_eeq_init > 0:
                scalar_stats.update({'init_eeq_loss': init_eeq_loss})
                loss += init_eeq_loss * weight_eeq_init
            if weight_eeq_coarse > 0:
                scalar_stats.update({'coarse_eeq_loss': coarse_eeq_loss})
                loss += coarse_eeq_loss * weight_eeq_coarse
            if weight_eeq_evolve > 0:
                scalar_stats.update({'evolve_eeq_loss': evolve_eeq_loss})
                loss += evolve_eeq_loss * weight_eeq_evolve

        ## region
        if self.cfg.model.with_rasterize_net:
            for py_name in ('init', 'coarse'):
                gt_masks = self._create_targets(
                    output[f'img_gt_{py_name}_polys'].clone().detach().cpu().numpy(),
                    [int(self.cfg.data.input_w / self.cfg.data.down_ratio),
                     int(self.cfg.data.input_h / self.cfg.data.down_ratio)])
                with torch.no_grad():
                    gt_masks = torch.from_numpy(gt_masks).to(output['pred_mask'][py_name].device).long()
                    gt_masks.requires_grad = False
                region_loss = self.region_crit(output['pred_mask'][py_name], gt_masks)
                if 'region' in self.weight_dict:
                    weight_region_crit = self.weight_dict['region'] * self.weight_dict[py_name]
                else:
                    weight_region_crit = self.weight_dict[py_name]

                scalar_stats.update({f'region_{py_name}_loss': region_loss})
                loss += region_loss * weight_region_crit

            for pyi in range(len(py_pred)):
                py_name = f'py{pyi}'
                gt_masks = self._create_targets(py_pred[pyi].clone().detach().cpu().numpy(),
                    [int(self.cfg.data.input_w / self.cfg.data.down_ratio),
                     int(self.cfg.data.input_h / self.cfg.data.down_ratio)])
                with torch.no_grad():
                    gt_masks = torch.from_numpy(gt_masks).to(output['pred_mask'][py_name].device).long()
                    gt_masks.requires_grad = False
                region_loss = self.region_crit(output['pred_mask'][py_name], gt_masks)
                if 'region' in self.weight_dict:
                    weight_region_crit = self.weight_dict['region'] * self.weight_dict['evolve']
                else:
                    weight_region_crit = self.weight_dict['evolve']
                scalar_stats.update({f'region_{py_name}_loss': region_loss})
                loss += region_loss * weight_region_crit

        if self.cfg.model.with_sharp_contour:
            if epoch >= self.cfg.train.sharp_param['ipc_start_epoch']:
                if 'ipc' in output:
                    n_iter_ipc = len(output['ipc'])
                    for i_ipc in range(n_iter_ipc):
                        ipc_loss = self.ipc_crit(output['ipc'][i_ipc], output['ipc_gt'][i_ipc])
                        scalar_stats.update({f'ipc_loss_{i_ipc}': ipc_loss})
                        loss += ipc_loss * self.weight_dict['ipc']

                if 'ipc_random' in output:
                    n_iter_ipc = len(output['ipc_random'])
                    ipc_loss_random = dummy
                    for i_ipc in range(n_iter_ipc):
                        ipc_loss_random += self.ipc_crit(output['ipc_random'][i_ipc],
                                                         output['ipc_gt_random'][i_ipc])
                    if self.cfg.train.sharp_param['avg_ipc_random_loss']:
                        ipc_loss_random /= n_iter_ipc
                    scalar_stats.update({f'ipc_random_loss(M={n_iter_ipc})': ipc_loss_random})
                    loss += ipc_loss_random * self.weight_dict['ipc']

            if self.cfg.train.sharp_param['train_with_refine'] and epoch >= self.cfg.train.sharp_param['refine_start_epoch']:
                if self.cfg.train.sharp_param['refine_with_dml'] and epoch >= self.start_epoch:
                    num_sharp_iter = len(output['py_pred']) - self.cfg.model.evolve_iters
                    refine_dm_loss = dummy
                    for i_sharp in range(self.cfg.model.evolve_iters, len(output['py_pred'])):
                        if num_polys == 0:
                            part_refine_dm_loss = dummy
                            len_refine_loss = dummy
                        else:
                            # 🚨 DDP-safe check: Ground Truth가 없는 경우를 처리합니다.
                            if len(output['img_gt_polys']) == 0:
                                part_refine_dm_loss = dummy
                            else:
                                part_refine_dm_loss = self.dml_crit(output['py_pred'][i_sharp - 1],
                                                                    output['py_pred'][i_sharp],
                                                                    output['img_gt_polys'],
                                                                    keyPointsMask)
                            len_refine_loss = torch.mean(torch.mean(torch.norm(output['py_pred'][i_sharp] - torch.roll(output['py_pred'][i_sharp], 1, -2), dim=-1), -1))
                        refine_dm_loss += (part_refine_dm_loss + len_refine_loss * self.weight_dict[
                            'length']) / num_sharp_iter
                        scalar_stats.update(
                            {'sharp_loss_{}'.format(i_sharp - self.cfg.model.evolve_iters): part_refine_dm_loss})
                        scalar_stats.update({f'len_sharp{i_sharp - self.cfg.model.evolve_iters}_loss': len_refine_loss})
                    loss += refine_dm_loss * self.weight_dict['sharp']
                else:
                    num_sharp_iter = len(output['py_pred']) - self.cfg.model.evolve_iters
                    refine_py_loss = dummy
                    for i_sharp in range(self.cfg.model.evolve_iters, len(output['py_pred'])):
                        if num_polys == 0:
                            part_py_loss = dummy
                            len_refine_loss = dummy
                        else:
                            part_py_loss = self.py_crit(output['py_pred'][i_sharp], output['img_gt_polys'])
                            len_refine_loss = torch.mean(
                                torch.mean(torch.norm(output['py_pred'][i_sharp] - torch.roll(
                                    output['py_pred'][i_sharp], 1, -2), dim=-1), -1))
                        refine_py_loss += (part_py_loss + len_refine_loss * self.weight_dict[
                            'length']) / num_sharp_iter
                        scalar_stats.update(
                            {'sharp_loss_{}'.format(i_sharp - self.cfg.model.evolve_iters): part_py_loss})
                        scalar_stats.update({f'len_sharp{i_sharp - self.cfg.model.evolve_iters}_loss': len_refine_loss})

                    loss += refine_py_loss * self.weight_dict['sharp']

        ## knowledge distillation
        if output_t is not None:
            # feature
            part = 'base'
            part_list = []
            for key_loss in self.cfg.train.kd_param['losses']:
                if f'ft_{part}' in key_loss:
                    part_list.append(key_loss)
            for each_kd_loss in part_list:
                pos_feature = int(each_kd_loss.split('_')[-1])
                kd_inter_loss = self.kd_inter_crit(output['feature_banks'][part][pos_feature], output_t['feature_banks'][part][pos_feature])
                scalar_stats.update({f'kd_{each_kd_loss}_loss': kd_inter_loss})
                if f'kd_{each_kd_loss}' in self.weight_dict:
                    weight_kd = self.weight_dict[f'kd_{each_kd_loss}']
                elif f'kd_ft_{part}' in self.weight_dict:
                    weight_kd = self.weight_dict[f'kd_ft_{part}']
                elif 'kd' in self.weight_dict:
                    weight_kd = self.weight_dict['kd']
                else:
                    weight_kd = 1.
                loss += weight_kd * kd_inter_loss

            for part in ['cnn_feature','feature_coarse']:
                if part in self.cfg.train.kd_param['losses']:
                    kd_inter_loss = self.kd_inter_crit(output[part], output_t[part])
                    scalar_stats.update({f'kd_{part}_loss': kd_inter_loss})
                    if f'kd_{part}' in self.weight_dict:
                        weight_kd = self.weight_dict[f'kd_{part}']
                    elif 'kd' in self.weight_dict:
                        weight_kd = self.weight_dict['kd']
                    else:
                        weight_kd = 1.
                    loss += weight_kd * kd_inter_loss
            # pixel
            part = 'pixel'
            if part in self.cfg.train.kd_param['losses']:
                target = pixel_gt.bool().long()
                kd_ct_loss = getattr(self, f"kd_{self.cfg.train.kd_param['losses'][part]}_crit")(
                    output[f'{part}'], output_t[f'{part}'], target)
                scalar_stats.update({f'kd_{part}_loss': kd_ct_loss})
                if f'kd_{part}' in self.weight_dict:
                    weight_kd = self.weight_dict[f'kd_{part}']
                elif 'kd' in self.weight_dict:
                    weight_kd = self.weight_dict['kd']
                else:
                    weight_kd = 1.
                if self.cfg.train.kd_param['weight_type'] == 'normalized':
                    if weight_kd >= 1.:
                        weight_kd /= (1 + weight_kd)
                    weight_kd *= self.weight_dict[f'{part}']
                loss += weight_kd * kd_ct_loss
            # center
            part = 'ct'
            if part in self.cfg.train.kd_param['losses']:
                kd_ct_loss = getattr(self, f"kd_{self.cfg.train.kd_param['losses'][part]}_crit")(output[f'{part}_hm'], output_t[f'{part}_hm'], batch[f'{part}_hm'])
                scalar_stats.update({f'kd_{part}_loss': kd_ct_loss})
                if f'kd_{part}' in self.weight_dict:
                    weight_kd = self.weight_dict[f'kd_{part}']
                elif 'kd' in self.weight_dict:
                    weight_kd = self.weight_dict['kd']
                else:
                    weight_kd = 1.
                if self.cfg.train.kd_param['weight_type'] == 'normalized':
                    if weight_kd >= 1.:
                        weight_kd /= (1 + weight_kd)
                    weight_kd *= self.weight_dict[f'box_{part}']
                loss += weight_kd * kd_ct_loss

            # init & coarse
            for part in ['init','coarse']:
                if part in self.cfg.train.kd_param['losses']:
                    kd_py_loss = getattr(self, f"kd_{self.cfg.train.kd_param['losses'][part]}_crit")(
                        output[f'poly_{part}'], output_t[f'poly_{part}'], output['img_gt_polys'])
                    scalar_stats.update({f'kd_py_loss_{part}': kd_py_loss})
                    if f'kd_{part}' in self.weight_dict:
                        weight_kd = self.weight_dict[f'kd_{part}']
                    elif 'kd' in self.weight_dict:
                        weight_kd = self.weight_dict['kd']
                    else:
                        weight_kd = 1.
                    if self.cfg.train.kd_param['weight_type']=='normalized':
                        if weight_kd >= 1.:
                            weight_kd /= (1 + weight_kd)
                        weight_kd *= self.weight_dict[part]
                    loss += weight_kd * kd_py_loss

            # evolve (snake)
            for i in range(self.cfg.model.evolve_iters):
                if f'evolve_{i}' in self.cfg.train.kd_param['losses']:
                    kd_py_loss = getattr(self,f"kd_{self.cfg.train.kd_param['losses'][f'evolve_{i}']}_crit")(output['py_pred'][i], output_t['py_pred'][i], output['img_gt_polys'])
                    scalar_stats.update({f'kd_py_loss_{i}': kd_py_loss})
                    if f'kd_py_{i}' in self.weight_dict:
                        weight_kd = self.weight_dict[f'kd_py_{i}']
                    elif 'kd' in self.weight_dict:
                        weight_kd = self.weight_dict['kd']
                    else:
                        weight_kd = 1.
                    if self.cfg.train.kd_param['weight_type']=='normalized':
                        if weight_kd >= 1.:
                            weight_kd /= (1 + weight_kd)
                        weight_kd *= self.weight_dict['evolve']
                    loss += weight_kd * kd_py_loss

        scalar_stats.update({'loss': loss})
        if not torch.isfinite(loss): #edit:debug:ddp-stop:25-08-09
            # 그래프 유지 + 0으로 클램프
            loss = (loss * 0.0) + dummy
            # 원인 찾으려면 로그 추가
            print(f"[R{dist.get_rank()}] non-finite loss at epoch={batch['epoch']} step?", flush=True)

        return output, loss, scalar_stats, out_ontraining

    @torch.no_grad()
    def _create_targets(self, instances, img_hw, lid=0):
        masks = []
        for obj_i in range(instances.shape[0]):
            instance = instances[obj_i].astype(np.int32)
            mask = np.zeros((img_hw[0], img_hw[1], 1), np.uint8)
            masks.append(cv2.fillPoly(mask, [instance], 1))
        masks = np.stack(masks, axis=0)  # (Nc, H, W, 1)
        return masks.squeeze(-1)
