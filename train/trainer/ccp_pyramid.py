import torch.nn as nn
from .utils import (FocalLoss, DMLoss, sigmoid, TVLoss, CurvLoss, MDLoss, mIoULoss, EdgeStandardDeviationLoss, BoundedRegLoss,
                    SoftCELoss, FocalCELoss, CosineSimLoss, SoftBCELoss, MeanSimLoss, CDLoss, VertexClsLoss,
                    _downsample_closed_poly, _resample_closed_poly_batch)
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

def _make_range(label: str, n_steps: int):
    if label == 'none': return []
    if label == 'final': return [n_steps - 1] if n_steps > 0 else []
    if label == 'penultimate': return [n_steps - 2] if n_steps > 1 else []
    if label == 'last2': return list(range(max(n_steps - 2, 0), n_steps))
    if label == 'except_1st': return list(range(1, n_steps))
    if label == 'all': return list(range(n_steps))
    # fallback
    return []


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
            else:
                pix_type = self.cfg.train.loss_type['pixel'] if 'pixel' in self.cfg.train.loss_type else 'ce'
                if pix_type == 'focal':
                    self.pix_crit = FocalCELoss(gamma=self.cfg.train.loss_params['pixel']['gamma'] if 'gamma' in self.cfg.train.loss_params['pixel'] else 2,
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

    def forward(self, batch, mode='default', output_t=None):
        # print(f"[NETWORK WRAPPER] meta={batch.get('meta')}")
        if 'test' in batch['meta']:
            output = self.net(batch['inp'], batch=batch)
            return output
        else:
            output = self.net(batch['inp'], batch=batch)
            return self.compute_loss(output, batch, output_t=output_t, mode=mode)
            # for k, v in output.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"[rank{dist.get_rank()}] {k}: shape={v.shape}, dtype={v.dtype}, requires_grad={v.requires_grad}")
            #     else:
            #         for elem in v:
            #             print(
            #                 f"[rank{dist.get_rank()}] {k}: shape={elem.shape}, dtype={elem.dtype}, requires_grad={elem.requires_grad}")
    def compute_loss(self, output, batch, output_t=None, mode='default'):
        out_ontraining = {}
        epoch = batch['epoch']
        scalar_stats = {}
        loss = 0.

        # 🚨 DDP-safe dummy loss:
        # 모델 파라미터에 연결된 0 값의 손실을 생성하여 DDP 환경에서 연산 그래프를 유지합니다.
        dummy_loss = 0.0 * sum(p.sum() for p in self.net.parameters())

        # ------------------------ 선택/단축 표기 ------------------------
        if self.cfg.model.with_img_idx:
            poly_init = output['poly_init'][batch['ct_01']]
            poly_coarse = output['poly_coarse'][batch['ct_01']]
            py_pred = [py[batch['ct_01']] for py in output['py_pred']]
            gt_full = output['img_gt_polys']  # (N, Vgt, 2) 예상
        else:
            poly_init = output['poly_init']
            poly_coarse = output['poly_coarse']
            py_pred = output['py_pred']
            gt_full = output['img_gt_polys']

        num_polys = poly_init.shape[0] if isinstance(poly_init, torch.Tensor) else 0
        n_evolve = len(py_pred)  # ✅ 동적 단계 수

        # ------------------------ Vertex Classification ------------------------
        keyPointsMask = batch['keypoints_mask'][batch['ct_01']]
        if self.cfg.train.weight_dict.get("vertex_cls", 0.) > 0:
            if 'py_valid_logits' in output and len(output['py_valid_logits']) > 0 and num_polys > 0:
                Vi_last = output['py_pred'][-1].shape[1]
                if 'img_gt_polys' in output and output['img_gt_polys'] is not None:
                    gt_last = _downsample_closed_poly(output['img_gt_polys'], Vi_last,
                                                      fallback_fn=_resample_closed_poly_batch)
                else:
                    gt_last = _resample_closed_poly_batch(output['img_gt_polys'], Vi_last)

                vertex_cls_loss = 0.0
                for py_valid_logit in output['py_valid_logits']:
                    pred_vertex_logits = py_valid_logit.permute(0, 2, 1)  # (B,N,2)
                    pred_coords = output['py_pred'][-1]
                    vertex_cls_loss += self.loss_dict["vertex_cls"](pred_vertex_logits, pred_coords,
                                                                    gt_last, keyPointsMask) / len(
                        output['py_valid_logits'])
            else:
                vertex_cls_loss = dummy_loss
            scalar_stats.update({'vtx_cls_loss': vertex_cls_loss})
            loss += self.cfg.train.weight_dict.get("vertex_cls", 0.) * vertex_cls_loss

        # ------------------------ Pixel / CT ------------------------
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
                else:
                    pix_loss = self.pix_crit(output['pixel'][pixelmap_i], pixel_gt.bool().long())
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
                ct_loss = dummy_loss
            else:
                # edit:ccp+ive-ct_loss:25-08-09
                pred_ct = output['ct_hm']  # (N,1,Hp,Wp)
                tgt_ct = batch['ct_hm']  # (N,1,Ht,Wt)
                if pred_ct.shape[-2:] != tgt_ct.shape[-2:]:
                    # GT를 예측 해상도에 맞춤 (heatmap은 nearest 권장)
                    tgt_ct = F.interpolate(
                        tgt_ct, size=pred_ct.shape[-2:], mode='nearest'
                    )

                ct_loss = self.ct_crit(torch.sigmoid(pred_ct), tgt_ct)
            # print(f"({dist.get_rank()})[DEBUG] ct_loss requires_grad:", ct_loss.requires_grad)
            scalar_stats.update({'ct_loss': ct_loss})
            loss += weight_ct * ct_loss

        # init & coarse
        if self.cfg.model.with_img_idx:
            poly_init = output['poly_init'][batch['ct_01']]
            poly_coarse = output['poly_coarse'][batch['ct_01']]
            py_pred = []
            for py in output['py_pred']:
                py_pred.append(py[batch['ct_01']])
        else:
            poly_init = output['poly_init']
            poly_coarse = output['poly_coarse']
            py_pred = output['py_pred']

        num_polys = len(poly_init)

        if num_polys == 0:
            init_py_loss = dummy_loss
            coarse_py_loss = dummy_loss
        else:
            # print(f"poly_init] :  {poly_init.max()}, output['img_gt_init_polys'] : {output['img_gt_init_polys'].max()}")
            init_py_loss = self.py_crit(poly_init, output['img_gt_init_polys'])
            coarse_py_loss = self.py_crit(poly_coarse, output['img_gt_coarse_polys'])

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
        if self.weight_dict['init'] > 0:
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

        # ------------------------ evolve (snake) ------------------------
            # range/시작 epoch를 동적으로 생성
            if self.ml_range_py:
                special_loss_range = _make_range(self.cfg.train.ml_range_py, n_evolve)
                special_loss_start_epoch = self.ml_start_epoch
                use_ml = True
            elif self.dml_range:
                special_loss_range = _make_range(self.cfg.train.dml_range, n_evolve)
                special_loss_start_epoch = self.dml_start_epoch
                use_ml = 'dml'
            elif self.mdml_range:
                special_loss_range = _make_range(self.cfg.train.mdml_range, n_evolve)
                special_loss_start_epoch = self.mdml_start_epoch
                use_ml = 'mdml'
            else:
                special_loss_range = []
                special_loss_start_epoch = 0
                use_ml = False

            if self.weight_dict['evolve'] > 0:
                n = n_evolve
                for i in range(n):
                    # KD 가중치 계산 (기존 로직 그대로)
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
                    else:
                        weight_py = self.weight_dict['evolve']

                    if num_polys == 0:
                        part_loss = dummy_loss
                    else:
                        Vi = py_pred[i].shape[1]
                        # 우선순위: img_gt_polys_hi(최대해상도 GT) -> 폴백(호길이 리샘플)
                        if 'img_gt_polys' in output and output['img_gt_polys'] is not None:
                            gtVi = _downsample_closed_poly(
                                output['img_gt_polys'], Vi,
                                fallback_fn=_resample_closed_poly_batch  # 폴백
                            )
                        else:
                            gtVi = _resample_closed_poly_batch(output['img_gt_polys'], Vi)

                        if (i in special_loss_range) and (epoch >= special_loss_start_epoch):
                            if use_ml == True and self.with_dml:
                                part_loss = self.ml_crit(py_pred[i - 1], py_pred[i], gtVi, keyPointsMask)
                            elif use_ml == 'mdml' and self.cfg.train.with_mdl:
                                part_loss = self.ml_crit(py_pred[i - 1], py_pred[i], gtVi).mean()
                            elif use_ml == 'dml':
                                part_loss = self.dml_crit(py_pred[i - 1], py_pred[i], gtVi, keyPointsMask)
                            else:
                                part_loss = self.py_crit(py_pred[i], gtVi)
                        else:
                            part_loss = self.py_crit(py_pred[i], gtVi)

                    scalar_stats.update({f'py_loss_{i}': part_loss})
                    loss += part_loss / max(n, 1) * weight_py

        ## total variation
        if ('tv' in self.weight_dict) or ('tv_coarse' in self.weight_dict) or ('tv_init' in self.weight_dict) or ('tv_evolve' in self.weight_dict):
            if num_polys == 0:
                init_tv_loss = dummy_loss
                coarse_tv_loss = dummy_loss
                evolve_tv_loss = dummy_loss
            else:
                init_tv_loss = self.tv_crit(poly_init)
                coarse_tv_loss = self.tv_crit(poly_coarse)
                evolve_tv_loss = 0.
                for i in range(len(py_pred)):
                    evolve_tv_loss += self.tv_crit(py_pred[i]) / len(py_pred)
            if 'tv_init' in self.weight_dict:
                weight_tv_init = self.weight_dict['tv_init']
            else:
                weight_tv_init = self.weight_dict['init'] * self.weight_dict['tv']
            if 'tv_coarse' in self.weight_dict:
                weight_tv_coarse = self.weight_dict['tv_coarse']
            else:
                weight_tv_coarse = self.weight_dict['coarse'] * self.weight_dict['tv']
            if 'tv_evolve' in self.weight_dict:
                weight_tv_evolve = self.weight_dict['tv_evolve']
            else:
                weight_tv_evolve = self.weight_dict['evolve'] * self.weight_dict['tv']
            if weight_tv_init > 0:
                scalar_stats.update({'init_tv_loss': init_tv_loss})
                loss += init_tv_loss * weight_tv_init
            if weight_tv_coarse > 0:
                scalar_stats.update({'coarse_tv_loss': coarse_tv_loss})
                loss += coarse_tv_loss * weight_tv_coarse
            if weight_tv_evolve > 0:
                scalar_stats.update({'evolve_tv_loss': evolve_tv_loss})
                loss += evolve_tv_loss * weight_tv_evolve

        if ('cv' in self.weight_dict) or ('cv_coarse' in self.weight_dict):
            if num_polys == 0:
                init_cv_loss = dummy_loss
                coarse_cv_loss = dummy_loss
                evolve_cv_loss = dummy_loss
            else:
                init_cv_loss = self.cv_crit(poly_init)
                coarse_cv_loss = self.cv_crit(poly_coarse)
                evolve_cv_loss = 0.
                for i in range(len(py_pred)):
                    evolve_cv_loss += self.cv_crit(py_pred[i]) / len(py_pred)
            if 'cv_init' in self.weight_dict:
                weight_cv_init = self.weight_dict['cv_init']
            else:
                weight_cv_init = self.weight_dict['init'] * self.weight_dict['cv']
            if 'cv_coarse' in self.weight_dict:
                weight_cv_coarse = self.weight_dict['cv_coarse']
            else:
                weight_cv_coarse = self.weight_dict['coarse'] * self.weight_dict['cv']
            if 'cv_evolve' in self.weight_dict:
                weight_cv_evolve = self.weight_dict['cv_evolve']
            else:
                weight_cv_evolve = self.weight_dict['evolve'] * self.weight_dict['cv']
            if weight_cv_init > 0:
                scalar_stats.update({'init_cv_loss': init_cv_loss})
                loss += init_cv_loss * weight_cv_init
            if weight_cv_coarse > 0:
                scalar_stats.update({'coarse_cv_loss': coarse_cv_loss})
                loss += coarse_cv_loss * weight_cv_coarse
            if weight_cv_evolve > 0:
                scalar_stats.update({'evolve_cv_loss': evolve_cv_loss})
                loss += evolve_cv_loss * weight_cv_evolve

        ## edge standard deviation loss (Edge Equal loss = eeq loss)
        if self.eeq_crit is not None:
            if num_polys == 0:
                init_eeq_loss = dummy_loss
                coarse_eeq_loss = dummy_loss
                evolve_eeq_loss = dummy_loss
            else:
                init_eeq_loss = self.eeq_crit(poly_init)
                coarse_eeq_loss = self.eeq_crit(poly_coarse)
                evolve_eeq_loss = 0.
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
            # for IPC
            if epoch >= self.cfg.train.sharp_param['ipc_start_epoch']:
                if 'ipc' in output:
                    n_iter_ipc = len(output['ipc'])
                    for i_ipc in range(n_iter_ipc):
                        ipc_loss = self.ipc_crit(output['ipc'][i_ipc], output['ipc_gt'][i_ipc])
                        scalar_stats.update({f'ipc_loss_{i_ipc}': ipc_loss})
                        loss += ipc_loss * self.weight_dict['ipc']

                if 'ipc_random' in output:
                    n_iter_ipc = len(output['ipc_random'])
                    ipc_loss_random = 0.
                    for i_ipc in range(n_iter_ipc):
                        ipc_loss_random += self.ipc_crit(output['ipc_random'][i_ipc],
                                                         output['ipc_gt_random'][i_ipc])
                    if self.cfg.train.sharp_param['avg_ipc_random_loss']:
                        ipc_loss_random /= n_iter_ipc
                    scalar_stats.update({f'ipc_random_loss(M={n_iter_ipc})': ipc_loss_random})
                    loss += ipc_loss_random * self.weight_dict['ipc']

            # for sharp refine
            if self.cfg.train.sharp_param['train_with_refine'] and epoch >= self.cfg.train.sharp_param['refine_start_epoch']:
                if self.cfg.train.sharp_param['refine_with_dml'] and epoch >= self.start_epoch:
                    num_sharp_iter = len(output['py_pred']) - self.cfg.model.evolve_iters
                    refine_dm_loss = 0.
                    for i_sharp in range(self.cfg.model.evolve_iters, len(output['py_pred'])):
                        if num_polys == 0:
                            part_refine_dm_loss = dummy_loss
                            len_refine_loss = dummy_loss
                        else:
                            # 🚨 DDP-safe check: Ground Truth가 없는 경우를 처리합니다.
                            if len(output['img_gt_polys']) == 0:
                                part_refine_dm_loss = dummy_loss
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
                    refine_py_loss = 0.
                    for i_sharp in range(self.cfg.model.evolve_iters, len(output['py_pred'])):
                        if num_polys == 0:
                            part_py_loss = dummy_loss
                            len_refine_loss = dummy_loss
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
                #edit:ccp+ive-ct_loss:25-08-09
                pred_ct = output[f'{part}_hm']
                tea_ct = output_t[f'{part}_hm']
                if pred_ct.shape[-2:] != tea_ct.shape[-2:]:
                    tea_ct = F.interpolate(tea_ct, size=pred_ct.shape[-2:], mode='nearest')
                kd_ct_loss = getattr(self, f"kd_{self.cfg.train.kd_param['losses'][part]}_crit")(pred_ct, tea_ct,
                                                                                                 batch[f'{part}_hm'])

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

            # --- KD evolve ---
            # evolve (snake)
            for i in range(len(py_pred)):  # 동적
                if f'evolve_{i}' in self.cfg.train.kd_param['losses']:
                    Vi = output['py_pred'][i].shape[1]
                    if 'img_gt_polys' in output and output['img_gt_polys'] is not None:
                        gtVi = _downsample_closed_poly(output['img_gt_polys'], Vi,
                                                       fallback_fn=_resample_closed_poly_batch)
                    else:
                        gtVi = _resample_closed_poly_batch(output['img_gt_polys'], Vi)

                    kd_py_loss = getattr(self, f"kd_{self.cfg.train.kd_param['losses'][f'evolve_{i}']}_crit")(
                        output['py_pred'][i], output_t['py_pred'][i], gtVi
                    )
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
        # 🚨 DDP 환경에서 연산 그래프를 끊어 교착 상태를 유발하므로 제거합니다.
        # output = safe_output(output)

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
