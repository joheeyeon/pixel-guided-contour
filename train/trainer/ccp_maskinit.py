import torch.nn as nn
from .utils import (FocalLoss, DMLoss, sigmoid, TVLoss, CurvLoss, MDLoss, mIoULoss, EdgeStandardDeviationLoss, BoundedRegLoss,
                    SoftCELoss, FocalCELoss, CosineSimLoss, SoftBCELoss, MeanSimLoss, CDLoss, VertexClsLoss,
                    DiceLoss, TverskyLoss, BoundaryLoss, ComboLoss, AdaptivePixelLoss)
import torch
import torch.nn.functional as F
import cv2, random, os
import numpy as np
import torch.distributed as dist


def visualize_pred_vs_gt(batch, output, cfg, epoch, save_dir="debug_vis/ccp_maskinit"):
    """
    GT와 Prediction의 모든 단계를 입력 이미지 위에 그려서 저장하는 디버깅용 함수.
    GT(초록색) 패널과 각 GCN 예측 단계(빨간색) 패널을 가로로 나열합니다.
    """
    # 분산 학습 환경에서는 메인 프로세스(rank 0)에서만 실행합니다.
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    # 시각화에 필요한 데이터 추출
    inp_imgs = batch['inp']
    gt_polys_all = output.get('img_gt_polys')
    py_pred_list = output.get('py_pred', [])
    batch_ind = output.get('batch_ind')

    if gt_polys_all is None or not py_pred_list or batch_ind is None or gt_polys_all.numel() == 0:
        return

    # 배치 내의 각 이미지를 순회하며 시각화
    for i in range(inp_imgs.size(0)):
        # 1. 원본 이미지 복원 (Un-normalize)
        mean = torch.tensor(cfg.data.mean, device=inp_imgs.device).view(3, 1, 1)
        std = torch.tensor(cfg.data.std, device=inp_imgs.device).view(3, 1, 1)
        img_tensor = inp_imgs[i] * std + mean
        img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        base_img_bgr = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        base_img_bgr = cv2.cvtColor(base_img_bgr, cv2.COLOR_RGB2BGR)
        H, W, _ = base_img_bgr.shape

        # 2. 현재 이미지에 해당하는 GT 인스턴스 필터링
        instance_indices = (batch_ind == i).nonzero(as_tuple=True)[0]
        if len(instance_indices) == 0:
            continue

        # 3. 패널 생성
        panels = []
        separator = np.full((H, 5, 3), 255, dtype=np.uint8)  # White separator

        # 3.1 GT 패널 (초록색)
        gt_polys_img = gt_polys_all[instance_indices].detach().cpu().numpy().astype(np.int32)
        gt_panel = base

        # 4. 이미지 저장
        # print(f"[DEBUG] len((batch['meta'].get('img_name', [img_{i}])) : {len(batch['meta'].get('img_name', [f'img_{i}']))}")
        img_name = f'img_{i}.png'
        save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{i}_{img_name}")
        cv2.imwrite(save_path, img_bgr)


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
        self.last_vis_epoch = -1  # 시각화를 위한 마지막 에폭 추적
        self.loss_dict = nn.ModuleDict()
        self.last_vis_epoch = -1

        if cfg.train.weight_dict.get("vertex_cls", 0.) > 0:
            self.loss_dict["vertex_cls"] = VertexClsLoss(**self.cfg.train.loss_params["vertex_cls"])

        if 'pixel' in self.cfg.model.heads:
            if self.cfg.model.heads['pixel'] == 1:
                # Binary pixel segmentation - use advanced loss functions
                pix_type = self.cfg.train.loss_type.get('pixel', 'focal') if hasattr(self.cfg.train, 'loss_type') else 'focal'
                pix_params = self.cfg.train.loss_params.get('pixel', {}) if hasattr(self.cfg.train, 'loss_params') else {}
                
                if pix_type == 'dice':
                    self.pix_crit = DiceLoss(smooth=pix_params.get('smooth', 1e-6))
                elif pix_type == 'tversky':
                    self.pix_crit = TverskyLoss(
                        alpha=pix_params.get('alpha', 0.7),
                        beta=pix_params.get('beta', 0.3),
                        smooth=pix_params.get('smooth', 1e-6)
                    )
                elif pix_type == 'boundary':
                    self.pix_crit = BoundaryLoss(
                        boundary_weight=pix_params.get('boundary_weight', 2.0),
                        smooth=pix_params.get('smooth', 1e-6)
                    )
                elif pix_type == 'combo':
                    self.pix_crit = ComboLoss(
                        focal_weight=pix_params.get('focal_weight', 0.5),
                        dice_weight=pix_params.get('dice_weight', 0.3),
                        tversky_weight=pix_params.get('tversky_weight', 0.2),
                        focal_alpha=pix_params.get('focal_alpha', 0.25),
                        focal_gamma=pix_params.get('focal_gamma', 2.0),
                        tversky_alpha=pix_params.get('tversky_alpha', 0.7)
                    )
                elif pix_type == 'adaptive':
                    self.pix_crit = AdaptivePixelLoss(
                        base_loss=pix_params.get('base_loss', 'focal'),
                        adaptive_weight=pix_params.get('adaptive_weight', True),
                        min_weight=pix_params.get('min_weight', 0.1),
                        max_weight=pix_params.get('max_weight', 10.0)
                    )
                else:  # 기본값: focal
                    self.pix_crit = FocalLoss()
            else:
                # Multi-class pixel segmentation
                pix_type = self.cfg.train.loss_type.get('pixel', 'ce')
                if pix_type == 'focal':
                    self.pix_crit = FocalCELoss(gamma=self.cfg.train.loss_params['pixel']['gamma'] if 'gamma' in self.cfg.train.loss_params['pixel'] else 2,
                                                reduce=self.cfg.train.loss_params['pixel']['reduce'] if 'reduce' in self.cfg.train.loss_params['pixel'] else True)
                else:
                    # Use deterministic-compatible cross entropy implementation
                    def deterministic_cross_entropy(input, target):
                        # Manual cross entropy implementation avoiding scatter/gather operations
                        # input: (B, C, H, W), target: (B, H, W)
                        B, C, H, W = input.shape
                        
                        # Compute log softmax manually (more stable)
                        max_vals = input.max(dim=1, keepdim=True)[0]
                        shifted = input - max_vals
                        exp_vals = torch.exp(shifted)
                        sum_exp = exp_vals.sum(dim=1, keepdim=True)
                        log_probs = shifted - torch.log(sum_exp)  # (B, C, H, W)
                        
                        # Create one-hot encoding for target without scatter operations
                        target_one_hot = torch.zeros_like(input)  # (B, C, H, W)
                        for c in range(C):
                            mask = (target == c)  # (B, H, W)
                            target_one_hot[:, c] = mask.float()
                        
                        # Compute cross entropy loss
                        loss = -(log_probs * target_one_hot).sum(dim=1).mean()
                        
                        return loss
                    
                    self.pix_crit = deterministic_cross_entropy

        self.ct_crit = FocalLoss()
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.tv_crit = TVLoss(type=cfg.train.loss_type.get('tv', 'smooth_l1') if cfg is not None else 'smooth_l1')
        self.cv_crit = CurvLoss(type=cfg.train.loss_type.get('cv', 'smooth_l1') if cfg is not None else 'smooth_l1')
        if self.cfg.model.with_rasterize_net:
            self.region_crit = mIoULoss(n_classes=2)

        if with_dml:
            self.ml_crit = DMLoss(type=cfg.train.loss_type.get('dm', 'smooth_l1') if cfg is not None else 'smooth_l1')
        elif cfg.train.with_mdl:
            self.ml_crit = MDLoss(type=cfg.train.loss_type.get('md', 'smooth_l1') if cfg is not None else 'smooth_l1',
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
            self.dml_crit = DMLoss(type=cfg.train.loss_type.get('dm', 'smooth_l1') if cfg is not None else 'smooth_l1')
        else:
            self.dml_range = []
        if self.cfg.train.mdml_range != 'none':
            self.mdml_range = PY_RANGE_DICT[self.cfg.train.mdml_range]
            self.mdml_crit = MDLoss(type=cfg.train.loss_type.get('md', 'smooth_l1') if cfg is not None else 'smooth_l1',
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
            self.kd_reg_crit = BoundedRegLoss(type=cfg.train.loss_type.get('kd_reg', 'smooth_l1'),
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
        # 항상 텐서+그래프로 시작 (rank/step 동일하게 존재), DDP-safe, edit:debug:ddp-stop:25-08-09
        loss = self._ddp_anchor * 0.0
        dummy = self._ddp_anchor * 0.0

        # keypoints_mask 가져오기 - 매칭된 순서로 재배열된 것이 있으면 사용
        if 'matched_keypoints_mask' in output:
            # 매칭된 순서로 재배열된 mask 사용
            keyPointsMask = output['matched_keypoints_mask']
        else:
            # 기존 방식 (매칭 실패 또는 비활성화된 경우)
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
                ct_loss = dummy
            else:
                ct_loss = self.ct_crit(sigmoid(output['ct_hm']), batch['ct_hm'])
            # print(f"({dist.get_rank()})[DEBUG] ct_loss requires_grad:", ct_loss.requires_grad)
            
            
            scalar_stats.update({'ct_loss': ct_loss})
            loss += weight_ct * ct_loss

        # init (remove: coarse)
        if self.cfg.model.with_img_idx:
            poly_init = output['poly_init'][batch['ct_01']]
        #     # poly_coarse = output['poly_coarse'][batch['ct_01']]
            py_pred = []
            for py in output['py_pred']:
                py_pred.append(py[batch['ct_01']])
        else:
            poly_init = output['poly_init']
        #     # poly_coarse = output['poly_coarse']
            py_pred = output['py_pred']
        #
        num_polys = len(poly_init)
        
        # GT-Prediction 매칭 시각화 비활성화 (성능 최적화)
        # if num_polys == 0:
        #     init_py_loss = dummy
        #     coarse_py_loss = dummy
        # else:
        #     # print(f"poly_init] :  {poly_init.max()}, output['img_gt_init_polys'] : {output['img_gt_init_polys'].max()}")
        #     init_py_loss = self.py_crit(poly_init, output['img_gt_init_polys'])
        #     # coarse_py_loss = self.py_crit(poly_coarse, output['img_gt_coarse_polys'])
        #
        # # print(f"({dist.get_rank()})[DEBUG] init_py_loss requires_grad:", init_py_loss.requires_grad)
        # # print(f"({dist.get_rank()})[DEBUG] coarse_py_loss requires_grad:", coarse_py_loss.requires_grad)
        # if (output_t is not None) and ('init' in self.cfg.train.kd_param['losses']) and (
        #         self.cfg.train.kd_param['weight_type'] == 'normalized'):
        #     if f'kd_init' in self.weight_dict:
        #         weight_kd = self.weight_dict[f'kd_init']
        #     elif 'kd' in self.weight_dict:
        #         weight_kd = self.weight_dict['kd']
        #     else:
        #         weight_kd = 0.5
        #     if weight_kd >= 1.:
        #         weight_kd = weight_kd / (1 + weight_kd)
        #     weight_py = self.weight_dict['init'] * (1 - weight_kd)
        #     print(f"weight_py (with kd) : {weight_py}")
        # else:
        #     weight_py = self.weight_dict['init']
        # if self.weight_dict['init'] > 0:
        #     scalar_stats.update({'init_py_loss': init_py_loss})
        #     loss += init_py_loss * weight_py

        # if (output_t is not None) and ('coarse' in self.cfg.train.kd_param['losses']) and (
        #         self.cfg.train.kd_param['weight_type'] == 'normalized'):
        #     if f'kd_coarse' in self.weight_dict:
        #         weight_kd = self.weight_dict[f'kd_coarse']
        #     elif 'kd' in self.weight_dict:
        #         weight_kd = self.weight_dict['kd']
        #     else:
        #         weight_kd = 0.5
        #     if weight_kd >= 1.:
        #         weight_kd = weight_kd / (1 + weight_kd)
        #     weight_py = self.weight_dict['coarse'] * (1 - weight_kd)
        #     print(f"weight_py (with kd) : {weight_py}")
        # else:
        #     weight_py = self.weight_dict['coarse']
        # if self.weight_dict['coarse'] > 0:
        #     scalar_stats.update({'coarse_py_loss': coarse_py_loss})
        #     loss += coarse_py_loss * weight_py

        # for snake (evolve)
        # print(f"[rank {torch.distributed.get_rank()}] 🚀 Starting py_loss computation")
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

        if self.weight_dict['evolve'] > 0:
            # py_loss = dummy
            n = len(py_pred)
            for i in range(1, n):
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
                    # ✅ Evolution step별 매칭 디버깅 (주석 처리)
                    # print(f"[DEBUG/STEP] py{i}: Pred center: {py_pred[i].mean(dim=-2)}")
                    
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

                # py_loss 디버깅 비활성화 (성능 최적화)
                
                scalar_stats.update({f'py_loss_{i}': part_loss})
                loss += part_loss / len(py_pred) * weight_py

        ## total variation
        if ('tv' in self.weight_dict) or ('tv_coarse' in self.weight_dict) or ('tv_init' in self.weight_dict) or ('tv_evolve' in self.weight_dict):
            if num_polys == 0:
                init_tv_loss = dummy
                # coarse_tv_loss = dummy
                evolve_tv_loss = dummy
            else:
                init_tv_loss = self.tv_crit(poly_init)
                # coarse_tv_loss = self.tv_crit(poly_coarse)
                evolve_tv_loss = dummy
                for i in range(len(py_pred)):
                    evolve_tv_loss += self.tv_crit(py_pred[i]) / len(py_pred)
            if 'tv_init' in self.weight_dict:
                weight_tv_init = self.weight_dict['tv_init']
            else:
                weight_tv_init = self.weight_dict['init'] * self.weight_dict['tv']
            # if 'tv_coarse' in self.weight_dict:
            #     weight_tv_coarse = self.weight_dict['tv_coarse']
            # else:
            #     weight_tv_coarse = self.weight_dict['coarse'] * self.weight_dict['tv']
            if 'tv_evolve' in self.weight_dict:
                weight_tv_evolve = self.weight_dict['tv_evolve']
            else:
                weight_tv_evolve = self.weight_dict['evolve'] * self.weight_dict['tv']
            if weight_tv_init > 0:
                scalar_stats.update({'init_tv_loss': init_tv_loss})
                loss += init_tv_loss * weight_tv_init
            # if weight_tv_coarse > 0:
            #     scalar_stats.update({'coarse_tv_loss': coarse_tv_loss})
            #     loss += coarse_tv_loss * weight_tv_coarse
            if weight_tv_evolve > 0:
                scalar_stats.update({'evolve_tv_loss': evolve_tv_loss})
                loss += evolve_tv_loss * weight_tv_evolve

        if ('cv' in self.weight_dict) or ('cv_coarse' in self.weight_dict):
            if num_polys == 0:
                init_cv_loss = dummy
                # coarse_cv_loss = dummy
                evolve_cv_loss = dummy
            else:
                init_cv_loss = self.cv_crit(poly_init)
                # coarse_cv_loss = self.cv_crit(poly_coarse)
                evolve_cv_loss = dummy
                for i in range(len(py_pred)):
                    evolve_cv_loss += self.cv_crit(py_pred[i]) / len(py_pred)
            if 'cv_init' in self.weight_dict:
                weight_cv_init = self.weight_dict['cv_init']
            else:
                weight_cv_init = self.weight_dict['init'] * self.weight_dict['cv']
            # if 'cv_coarse' in self.weight_dict:
            #     weight_cv_coarse = self.weight_dict['cv_coarse']
            # else:
            #     weight_cv_coarse = self.weight_dict['coarse'] * self.weight_dict['cv']
            if 'cv_evolve' in self.weight_dict:
                weight_cv_evolve = self.weight_dict['cv_evolve']
            else:
                weight_cv_evolve = self.weight_dict['evolve'] * self.weight_dict['cv']
            if weight_cv_init > 0:
                scalar_stats.update({'init_cv_loss': init_cv_loss})
                loss += init_cv_loss * weight_cv_init
            # if weight_cv_coarse > 0:
            #     scalar_stats.update({'coarse_cv_loss': coarse_cv_loss})
            #     loss += coarse_cv_loss * weight_cv_coarse
            if weight_cv_evolve > 0:
                scalar_stats.update({'evolve_cv_loss': evolve_cv_loss})
                loss += evolve_cv_loss * weight_cv_evolve

        ## edge standard deviation loss (Edge Equal loss = eeq loss)
        if self.eeq_crit is not None:
            if num_polys == 0:
                init_eeq_loss = dummy
                # coarse_eeq_loss = dummy
                evolve_eeq_loss = dummy
            else:
                init_eeq_loss = self.eeq_crit(poly_init)
                # coarse_eeq_loss = self.eeq_crit(poly_coarse)
                evolve_eeq_loss = dummy
                for i in range(len(py_pred)):
                    evolve_eeq_loss += self.eeq_crit(py_pred[i]) / len(py_pred)

            if 'edge_std_init' in self.weight_dict:
                weight_eeq_init = self.weight_dict['edge_std_init']
            else:
                weight_eeq_init = self.weight_dict['init'] * self.weight_dict['edge_std']
            # if 'edge_std_coarse' in self.weight_dict:
            #     weight_eeq_coarse = self.weight_dict['edge_std_coarse']
            # else:
            #     weight_eeq_coarse = self.weight_dict['coarse'] * self.weight_dict['edge_std']
            if 'edge_std_evolve' in self.weight_dict:
                weight_eeq_evolve = self.weight_dict['edge_std_evolve']
            else:
                weight_eeq_evolve = self.weight_dict['evolve'] * self.weight_dict['edge_std']

            if weight_eeq_init > 0:
                scalar_stats.update({'init_eeq_loss': init_eeq_loss})
                loss += init_eeq_loss * weight_eeq_init
            # if weight_eeq_coarse > 0:
            #     scalar_stats.update({'coarse_eeq_loss': coarse_eeq_loss})
            #     loss += coarse_eeq_loss * weight_eeq_coarse
            if weight_eeq_evolve > 0:
                scalar_stats.update({'evolve_eeq_loss': evolve_eeq_loss})
                loss += evolve_eeq_loss * weight_eeq_evolve

        ## region
        if self.cfg.model.with_rasterize_net:
            # for py_name in ('init', 'coarse'):
            for py_name in ('init', ):
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
                    ipc_loss_random = dummy
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

            # for part in ['cnn_feature','feature_coarse']:
            for part in ['cnn_feature']:
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
            # for part in ['init', 'coarse']:
            for part in ['init']:
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

        # --- Debug Visualization ---  <--- 여기부터 추가합니다.
        # 분산 학습 환경에서는 메인 프로세스(rank 0)에서만 실행합니다.
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main_process and getattr(self.cfg.train, 'debug_vis', False):
            vis_interval = getattr(self.cfg.train, 'vis_epoch_interval', 1)
            # Visualize only for designated epochs. To avoid visualizing on every step, we track the last visualized epoch.
            # if epoch % vis_interval == 0 and self.last_vis_epoch != epoch:
            print(f"{self.cfg.commen.result_dir}/debug")
            visualize_pred_vs_gt(batch, output, self.cfg, epoch, save_dir=f"{self.cfg.commen.result_dir}/debug")
            self.last_vis_epoch = epoch

        return output, loss, scalar_stats, out_ontraining
    
    def _log_gt_prediction_matching(self, poly_init, img_gt_polys, py_pred, epoch, num_polys, batch_idx=0, output=None, batch=None):
        """
        GT-Prediction 매칭 상태 로깅 및 저장
        """
        import torch
        import numpy as np
        import os
        import json
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        # 매칭 개수 정보
        num_gt = len(img_gt_polys) if img_gt_polys is not None else 0
        num_pred = len(poly_init) if poly_init is not None else 0
        
        print(f"\n[MATCH LOG] Epoch {epoch}, Batch {batch_idx}")
        print(f"  GT polygons: {num_gt}, Predicted polygons: {num_pred}")
        
        # ✅ 원본 데이터 확인
        if batch is not None and 'img_gt_polys' in batch:
            original_gt_shape = batch['img_gt_polys'].shape
            ct_01_shape = batch.get('ct_01', torch.tensor([])).shape
            print(f"  [DEBUG] Original GT shape: {original_gt_shape}, ct_01 shape: {ct_01_shape}")
            if 'ct_01' in batch:
                valid_gt_count = batch['ct_01'].sum().item()
                print(f"  [DEBUG] Valid GT count (ct_01): {valid_gt_count}")
        
        if output is not None and 'batch_ind' in output:
            batch_ind_shape = output['batch_ind'].shape
            unique_imgs = torch.unique(output['batch_ind']).tolist()
            print(f"  [DEBUG] batch_ind shape: {batch_ind_shape}, unique images: {unique_imgs}")
            print(f"  [DEBUG] Predictions per image: {[(img_idx, (output['batch_ind'] == img_idx).sum().item()) for img_idx in unique_imgs]}")
        
        # 저장 디렉토리 생성 (cfg.commen.result_dir 아래)
        log_dir = os.path.join(self.cfg.commen.result_dir, "gt_pred_matching")
        os.makedirs(log_dir, exist_ok=True)
        
        # 매칭 정보를 저장할 데이터
        matching_data = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "timestamp": datetime.now().isoformat(),
            "num_gt": num_gt,
            "num_pred": num_pred,
            "matching_info": []
        }
        
        if num_gt > 0 and num_pred > 0:
            # GT와 Prediction center 비교
            if isinstance(img_gt_polys, torch.Tensor) and img_gt_polys.numel() > 0:
                gt_centers = img_gt_polys.mean(dim=-2)  # (N, 2)
                gt_centers_np = gt_centers.detach().cpu().numpy()
                print(f"  GT centers: {gt_centers_np[:3]}")  # 첫 3개만
            
            if isinstance(poly_init, torch.Tensor) and poly_init.numel() > 0:
                pred_centers = poly_init.mean(dim=-2)  # (N, 2)
                pred_centers_np = pred_centers.detach().cpu().numpy()
                print(f"  Pred centers: {pred_centers_np[:3]}")  # 첫 3개만
                
                # 실제 최적 매칭 계산 (Hungarian algorithm 대신 간단한 최단거리 매칭)
                if isinstance(img_gt_polys, torch.Tensor) and len(gt_centers) > 0 and len(pred_centers) > 0:
                    min_matched = min(len(gt_centers), len(pred_centers))
                    
                    # 모든 GT-Pred 쌍의 거리 계산
                    distances = torch.cdist(gt_centers, pred_centers)  # (num_gt, num_pred)
                    
                    # ✅ Hungarian Algorithm을 사용한 최적 매칭
                    max_distance = getattr(self.cfg.train, 'max_center_distance_maskinit', 50.0)
                    
                    # 거리 임계값을 넘는 쌍들은 매우 큰 값으로 설정
                    cost_matrix = distances.clone()
                    cost_matrix[cost_matrix > max_distance] = 1e6
                    
                    # Hungarian algorithm 구현 (scipy 없이 간단한 버전)
                    optimal_matches = self._hungarian_matching(cost_matrix.cpu().numpy(), max_distance)
                    
                    # 매칭 통계
                    num_gt_total = len(gt_centers)
                    num_pred_total = len(pred_centers)
                    num_matched = len(optimal_matches)
                    num_unmatched_gt = num_gt_total - num_matched
                    num_unmatched_pred = num_pred_total - num_matched
                    
                    # 최적 매칭 결과 출력 및 저장
                    print(f"  Optimal matching within {max_distance:.1f} pixels:")
                    print(f"    Matched pairs: {num_matched}/{min(num_gt_total, num_pred_total)}")
                    print(f"    Unmatched GT: {num_unmatched_gt}, Unmatched Pred: {num_unmatched_pred}")
                    
                    # ✅ 매칭 실패 원인 분석
                    if num_unmatched_gt > 0 or num_unmatched_pred > 0:
                        print(f"  [DEBUG] Distance matrix shape: {distances.shape}")
                        print(f"  [DEBUG] All distances (min={distances.min():.2f}, max={distances.max():.2f})")
                        
                        # 매칭되지 않은 GT들과 가장 가까운 Pred 거리 확인
                        unmatched_gt_indices = set(range(num_gt_total)) - {match[0] for match in optimal_matches}
                        unmatched_pred_indices = set(range(num_pred_total)) - {match[1] for match in optimal_matches}
                        
                        for gt_idx in list(unmatched_gt_indices)[:3]:  # 처음 3개만
                            min_dist_to_pred = distances[gt_idx].min().item()
                            closest_pred = distances[gt_idx].argmin().item()
                            print(f"  [DEBUG] Unmatched GT{gt_idx}: closest Pred{closest_pred} at distance {min_dist_to_pred:.2f}")
                            
                        for pred_idx in list(unmatched_pred_indices)[:3]:  # 처음 3개만
                            min_dist_to_gt = distances[:, pred_idx].min().item()
                            closest_gt = distances[:, pred_idx].argmin().item()
                            print(f"  [DEBUG] Unmatched Pred{pred_idx}: closest GT{closest_gt} at distance {min_dist_to_gt:.2f}")
                    
                    if num_matched > 0:
                        for i, (gt_idx, pred_idx, distance) in enumerate(optimal_matches):
                            if i < 3:  # 첫 3개만 콘솔 출력
                                print(f"    GT{gt_idx} <-> Pred{pred_idx}, distance={distance:.2f}")
                            
                            # JSON에 저장할 데이터
                            matching_data["matching_info"].append({
                                "gt_idx": gt_idx,
                                "pred_idx": pred_idx,
                                "gt_center": gt_centers_np[gt_idx].tolist(),
                                "pred_center": pred_centers_np[pred_idx].tolist(),
                                "center_distance": distance
                            })
                        
                        # 매칭 품질 통계
                        optimal_distances = [match[2] for match in optimal_matches]
                        print(f"    Avg distance: {np.mean(optimal_distances):.2f}, Max: {np.max(optimal_distances):.2f}")
                    else:
                        print(f"    No matches found within {max_distance:.1f} pixel threshold!")
                    
                    min_matched = len(optimal_matches)
                    
                    # 시각화 생성 (처음 몇 개 polygon만)
                    if epoch < 3 or epoch % 10 == 0:  # epoch 0,1,2 또는 10의 배수일 때만
                        self._save_matching_visualization_by_image(
                            img_gt_polys, poly_init, gt_centers_np, pred_centers_np,
                            epoch, batch_idx, log_dir, optimal_matches, 
                            output.get('batch_ind'), batch
                        )
        
        # Evolution step별 정보
        if py_pred and len(py_pred) > 1:
            print(f"  Evolution steps: {len(py_pred)}")
            evolution_info = []
            for step_idx in range(min(2, len(py_pred))):  # 첫 2 step만
                py_step = py_pred[step_idx]
                if isinstance(py_step, torch.Tensor) and py_step.numel() > 0:
                    step_centers = py_step.mean(dim=-2)
                    step_centers_np = step_centers.detach().cpu().numpy()
                    print(f"    Step {step_idx} centers: {step_centers_np[:2]}")  # 첫 2개만
                    evolution_info.append({
                        "step": step_idx,
                        "centers": step_centers_np[:3].tolist()  # 첫 3개만 저장
                    })
            matching_data["evolution_info"] = evolution_info
        
        # Vertex alignment 정보 (최적 매칭된 첫 번째 쌍에 대해서만)
        vertex_alignment_info = []
        if num_gt > 0 and num_pred > 0 and 'optimal_matches' in locals() and len(optimal_matches) > 0:
            if isinstance(img_gt_polys, torch.Tensor) and isinstance(poly_init, torch.Tensor):
                # 최적 매칭된 첫 번째 쌍의 인덱스 가져오기
                gt_idx, pred_idx, match_distance = optimal_matches[0]
                
                if gt_idx < len(img_gt_polys) and pred_idx < len(poly_init):
                    print(f"  Best matched pair vertex alignment (GT{gt_idx} <-> Pred{pred_idx}, dist={match_distance:.2f}):")
                    
                    # vertex 0-3 비교 (alignment 확인용)
                    min_vertices = min(4, len(img_gt_polys[gt_idx]), len(poly_init[pred_idx]))
                    for v_idx in range(min_vertices):
                        gt_v = img_gt_polys[gt_idx][v_idx].detach().cpu().numpy()
                        pred_v = poly_init[pred_idx][v_idx].detach().cpu().numpy()
                        v_dist = np.linalg.norm(gt_v - pred_v)
                        print(f"    V{v_idx}: GT[{gt_v[0]:.1f},{gt_v[1]:.1f}] vs Pred[{pred_v[0]:.1f},{pred_v[1]:.1f}], dist={v_dist:.2f}")
                        vertex_alignment_info.append({
                            "gt_idx": gt_idx,
                            "pred_idx": pred_idx,
                            "vertex_idx": v_idx,
                            "gt_vertex": gt_v.tolist(),
                            "pred_vertex": pred_v.tolist(),
                            "vertex_distance": float(v_dist)
                        })
        matching_data["vertex_alignment"] = vertex_alignment_info
        
        # JSON 파일로 저장
        json_filename = f"matching_epoch{epoch:03d}_batch{batch_idx:03d}.json"
        json_path = os.path.join(log_dir, json_filename)
        try:
            with open(json_path, 'w') as f:
                json.dump(matching_data, f, indent=2)
        except Exception as e:
            print(f"  Warning: Could not save JSON log: {e}")
    
    def _save_matching_visualization(self, img_gt_polys, poly_init, gt_centers, pred_centers, 
                                   epoch, batch_idx, log_dir, optimal_matches):
        """GT-Prediction 최적 매칭 시각화 저장"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # Figure 설정
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            gt_polys_np = img_gt_polys.detach().cpu().numpy()
            pred_polys_np = poly_init.detach().cpu().numpy()
            
            # 매칭되지 않은 GT polygons (회색)
            matched_gt_indices = {match[0] for match in optimal_matches}
            for i, gt_poly in enumerate(gt_polys_np):
                if i not in matched_gt_indices:
                    polygon = patches.Polygon(gt_poly, fill=False, edgecolor='gray', linewidth=1, 
                                            linestyle='-', alpha=0.5)
                    ax.add_patch(polygon)
                    ax.plot(gt_centers[i][0], gt_centers[i][1], 'o', color='gray', markersize=6)
                    ax.text(gt_centers[i][0]+2, gt_centers[i][1]+2, f'GT{i}', color='gray', fontsize=8)
            
            # 매칭되지 않은 Prediction polygons (회색)
            matched_pred_indices = {match[1] for match in optimal_matches}
            for i, pred_poly in enumerate(pred_polys_np):
                if i not in matched_pred_indices:
                    polygon = patches.Polygon(pred_poly, fill=False, edgecolor='gray', linewidth=1,
                                            linestyle='--', alpha=0.5)
                    ax.add_patch(polygon)
                    ax.plot(pred_centers[i][0], pred_centers[i][1], 'o', color='gray', markersize=6)
                    ax.text(pred_centers[i][0]+2, pred_centers[i][1]+2, f'P{i}', color='gray', fontsize=8)
            
            # 최적 매칭된 쌍들 그리기
            for match_idx, (gt_idx, pred_idx, distance) in enumerate(optimal_matches[:10]):  # 최대 10개만
                # GT polygon (빨간색)
                gt_poly = gt_polys_np[gt_idx]
                polygon = patches.Polygon(gt_poly, fill=False, edgecolor='red', linewidth=2, 
                                        linestyle='-', label=f'GT' if match_idx == 0 else "")
                ax.add_patch(polygon)
                ax.plot(gt_centers[gt_idx][0], gt_centers[gt_idx][1], 'ro', markersize=8)
                ax.text(gt_centers[gt_idx][0]+2, gt_centers[gt_idx][1]+2, f'GT{gt_idx}', color='red', fontsize=10)
                
                # Prediction polygon (파란색)
                pred_poly = pred_polys_np[pred_idx]
                polygon = patches.Polygon(pred_poly, fill=False, edgecolor='blue', linewidth=2,
                                        linestyle='--', label=f'Pred' if match_idx == 0 else "")
                ax.add_patch(polygon)
                ax.plot(pred_centers[pred_idx][0], pred_centers[pred_idx][1], 'bo', markersize=8)
                ax.text(pred_centers[pred_idx][0]+2, pred_centers[pred_idx][1]+2, f'P{pred_idx}', color='blue', fontsize=10)
                
                # 매칭 라인 그리기
                ax.plot([gt_centers[gt_idx][0], pred_centers[pred_idx][0]], 
                       [gt_centers[gt_idx][1], pred_centers[pred_idx][1]], 
                       'green', linestyle=':', alpha=0.8, linewidth=2)
                
                # 거리 텍스트 표시 (첫 5개만)
                if match_idx < 5:
                    mid_x = (gt_centers[gt_idx][0] + pred_centers[pred_idx][0]) / 2
                    mid_y = (gt_centers[gt_idx][1] + pred_centers[pred_idx][1]) / 2
                    ax.text(mid_x, mid_y, f'{distance:.1f}', color='green', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            ax.set_xlim(0, 416)  # 이미지 크기에 맞게 조정
            ax.set_ylim(0, 416)
            ax.invert_yaxis()  # 이미지 좌표계에 맞게
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'GT-Prediction Matching (Epoch {epoch}, Batch {batch_idx})')
            ax.legend(loc='upper right')
            
            # 저장
            img_filename = f"matching_epoch{epoch:03d}_batch{batch_idx:03d}.png"
            img_path = os.path.join(log_dir, img_filename)
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  Warning: Could not save visualization: {e}")

    @torch.no_grad()
    def _save_matching_visualization_by_image(self, img_gt_polys, poly_init, gt_centers, pred_centers, 
                                            epoch, batch_idx, log_dir, optimal_matches, batch_ind, batch, img_idx=0):
        """이미지별로 GT-Prediction 매칭 시각화 저장 (간단한 버전) - gradient 계산에 영향 없음"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
            
            # 간단하게 첫 번째 이미지만 저장
            gt_polys_np = img_gt_polys.detach().cpu().numpy()
            pred_polys_np = poly_init.detach().cpu().numpy()
            gt_centers_np = gt_centers.detach().cpu().numpy() 
            pred_centers_np = pred_centers.detach().cpu().numpy()
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # 매칭된 쌍들 그리기 (같은 색상으로)
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan', 'magenta']
            
            print(f"  [DEBUG] Saving visualization with {len(optimal_matches)} matches")
            
            for match_idx, (gt_idx, pred_idx, distance) in enumerate(optimal_matches[:5]):  # 최대 5개만
                color = colors[match_idx % len(colors)]
                
                # GT polygon (실선)
                if gt_idx < len(gt_polys_np):
                    polygon = patches.Polygon(gt_polys_np[gt_idx], fill=False, edgecolor=color, linewidth=2, 
                                            linestyle='-', label=f'GT' if match_idx == 0 else "")
                    ax.add_patch(polygon)
                    ax.plot(gt_centers_np[gt_idx][0], gt_centers_np[gt_idx][1], 'o', color=color, markersize=8)
                    ax.text(gt_centers_np[gt_idx][0]+3, gt_centers_np[gt_idx][1]+3, f'GT{gt_idx}', color=color, fontsize=10, weight='bold')
                
                # Prediction polygon (점선, 같은 색상)
                if pred_idx < len(pred_polys_np):
                    polygon = patches.Polygon(pred_polys_np[pred_idx], fill=False, edgecolor=color, linewidth=2,
                                            linestyle='--', label=f'Pred' if match_idx == 0 else "")
                    ax.add_patch(polygon)
                    ax.plot(pred_centers_np[pred_idx][0], pred_centers_np[pred_idx][1], 's', color=color, markersize=8)
                    ax.text(pred_centers_np[pred_idx][0]+3, pred_centers_np[pred_idx][1]+3, f'P{pred_idx}', color=color, fontsize=10, weight='bold')
                    
                    # 매칭 라인 그리기
                    ax.plot([gt_centers_np[gt_idx][0], pred_centers_np[pred_idx][0]], 
                           [gt_centers_np[gt_idx][1], pred_centers_np[pred_idx][1]], 
                           color=color, linestyle=':', alpha=0.8, linewidth=2)
                    
                    # 거리 텍스트 표시
                    mid_x = (gt_centers_np[gt_idx][0] + pred_centers_np[pred_idx][0]) / 2
                    mid_y = (gt_centers_np[gt_idx][1] + pred_centers_np[pred_idx][1]) / 2
                    ax.text(mid_x, mid_y, f'{distance:.1f}', color=color, fontsize=9, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            ax.set_xlim(0, 416)
            ax.set_ylim(0, 416)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'GT-Pred Matching (Epoch {epoch}, Batch {batch_idx})\n'
                       f'Matches: {len(optimal_matches)}, GT: {len(gt_polys_np)}, Pred: {len(pred_polys_np)}')
            ax.legend(loc='upper right')
            
            # 저장
            img_filename = f"matching_epoch{epoch:03d}_batch{batch_idx:03d}_img{img_idx:02d}.png"
            img_path = os.path.join(log_dir, img_filename)
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  [DEBUG] Saved visualization to: {img_path}")
                
        except Exception as e:
            print(f"  Warning: Could not save image-wise visualization: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def _save_vertex_matching_visualization(self, gt_polys_img, pred_polys_img, matches, 
                                           epoch, batch_idx, log_dir, img_idx=0):
        """매칭된 contour들의 vertex별 정렬 상황 시각화 - gradient 계산에 영향 없음"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
            
            gt_polys_np = gt_polys_img.detach().cpu().numpy()
            pred_polys_np = pred_polys_img.detach().cpu().numpy()
            
            # 매칭된 첫 3개 쌍에 대해 vertex 시각화
            num_visualize = min(3, len(matches))
            if num_visualize == 0:
                return
            
            fig, axes = plt.subplots(1, num_visualize, figsize=(6*num_visualize, 6))
            if num_visualize == 1:
                axes = [axes]
            
            colors = ['red', 'blue', 'green']
            
            for match_idx in range(num_visualize):
                gt_idx, pred_idx, distance = matches[match_idx]
                ax = axes[match_idx]
                color = colors[match_idx]
                
                if gt_idx >= len(gt_polys_np) or pred_idx >= len(pred_polys_np):
                    continue
                
                gt_poly = gt_polys_np[gt_idx]
                pred_poly_aligned = pred_polys_np[pred_idx]  # 이미 poly_init에서 정렬됨
                
                # GT contour 그리기 (실선)
                polygon = patches.Polygon(gt_poly, fill=False, edgecolor=color, 
                                        linewidth=2, linestyle='-', alpha=0.8, label='GT')
                ax.add_patch(polygon)
                
                # Prediction contour 그리기 (점선) - 정렬된 버전
                polygon = patches.Polygon(pred_poly_aligned, fill=False, edgecolor=color,
                                        linewidth=2, linestyle='--', alpha=0.8, label='Pred')
                ax.add_patch(polygon)
                
                # Vertex 번호 표시 및 매칭 라인
                num_vertices = min(len(gt_poly), len(pred_poly_aligned))
                for v_idx in range(num_vertices):
                    gt_v = gt_poly[v_idx]
                    pred_v = pred_poly_aligned[v_idx]
                    
                    # GT vertex (원형 마커)
                    ax.plot(gt_v[0], gt_v[1], 'o', color=color, markersize=8, 
                           markeredgecolor='white', markeredgewidth=1)
                    ax.text(gt_v[0]+2, gt_v[1]+2, f'G{v_idx}', color=color, 
                           fontsize=8, weight='bold')
                    
                    # Pred vertex (사각형 마커)
                    ax.plot(pred_v[0], pred_v[1], 's', color=color, markersize=8,
                           markeredgecolor='white', markeredgewidth=1)
                    ax.text(pred_v[0]+2, pred_v[1]+2, f'P{v_idx}', color=color,
                           fontsize=8, weight='bold')
                    
                    # Vertex 매칭 라인
                    ax.plot([gt_v[0], pred_v[0]], [gt_v[1], pred_v[1]], 
                           color=color, linestyle=':', alpha=0.6, linewidth=1)
                    
                    # Vertex 거리 표시
                    v_dist = np.linalg.norm(gt_v - pred_v)
                    mid_x = (gt_v[0] + pred_v[0]) / 2
                    mid_y = (gt_v[1] + pred_v[1]) / 2
                    ax.text(mid_x, mid_y, f'{v_dist:.1f}', color=color, fontsize=7,
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
                
                # 축 설정
                ax.set_xlim(min(gt_poly[:, 0].min(), pred_poly_aligned[:, 0].min()) - 20,
                           max(gt_poly[:, 0].max(), pred_poly_aligned[:, 0].max()) + 20)
                ax.set_ylim(min(gt_poly[:, 1].min(), pred_poly_aligned[:, 1].min()) - 20,
                           max(gt_poly[:, 1].max(), pred_poly_aligned[:, 1].max()) + 20)
                ax.invert_yaxis()
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f'GT{gt_idx} ↔ Pred{pred_idx}\nCenter Dist: {distance:.2f}px')
                ax.legend()
            
            plt.suptitle(f'Vertex Matching (Epoch {epoch}, Batch {batch_idx}, Image {img_idx})')
            plt.tight_layout()
            
            # 저장
            vertex_filename = f"vertex_matching_epoch{epoch:03d}_batch{batch_idx:03d}_img{img_idx:02d}.png"
            vertex_path = os.path.join(log_dir, vertex_filename)
            plt.savefig(vertex_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  [DEBUG] Saved vertex visualization to: {vertex_path}")
            
        except Exception as e:
            print(f"  Warning: Could not save vertex visualization: {e}")
            import traceback
            traceback.print_exc()

    def _align_vertex_direction(self, gt_poly, pred_poly):
        """
        GT vertex 0을 기준으로 prediction vertex들을 올바른 방향으로 재정렬
        """
        import torch
        import numpy as np
        
        if len(gt_poly) != len(pred_poly):
            return pred_poly  # 길이가 다르면 그대로 반환
        
        gt_poly_np = gt_poly.detach().cpu().numpy() if torch.is_tensor(gt_poly) else gt_poly
        pred_poly_np = pred_poly.detach().cpu().numpy() if torch.is_tensor(pred_poly) else pred_poly
        
        # GT vertex 0과 가장 가까운 pred vertex 찾기
        gt_v0 = gt_poly_np[0]
        distances_to_v0 = [np.linalg.norm(pred_v - gt_v0) for pred_v in pred_poly_np]
        best_start_idx = np.argmin(distances_to_v0)
        
        # Forward 방향으로만 정렬 (GT가 일관된 방향으로 정렬되어 있음)
        best_reordered = np.roll(pred_poly_np, -best_start_idx, axis=0)
        avg_dist = np.mean([np.linalg.norm(gt_poly_np[i] - best_reordered[i]) for i in range(len(gt_poly_np))])
        
        print(f"    [VERTEX ALIGN] Start idx: {best_start_idx}, Forward distance: {avg_dist:.2f}")
        
        # 원래 타입으로 복원
        if torch.is_tensor(pred_poly):
            return torch.from_numpy(best_reordered).to(pred_poly.device).type(pred_poly.dtype)
        else:
            return best_reordered

    def _greedy_matching(self, distance_matrix, max_distance):
        """
        빠른 greedy 매칭 (Hungarian보다 훨씬 빠름)
        """
        import numpy as np
        
        matches = []
        used_gt = set()
        used_pred = set()
        
        # 거리 순으로 정렬
        num_gt, num_pred = distance_matrix.shape
        candidates = []
        
        for i in range(num_gt):
            for j in range(num_pred):
                if distance_matrix[i, j] <= max_distance:
                    candidates.append((distance_matrix[i, j], i, j))
        
        candidates.sort(key=lambda x: x[0])  # 거리 순 정렬
        
        for distance, gt_idx, pred_idx in candidates:
            if gt_idx not in used_gt and pred_idx not in used_pred:
                matches.append((gt_idx, pred_idx, distance))
                used_gt.add(gt_idx)
                used_pred.add(pred_idx)
        
        return matches

    def _hungarian_matching(self, cost_matrix, max_distance):
        """
        간단한 Hungarian Algorithm 구현 (scipy 없이)
        """
        import numpy as np
        
        # 정사각 행렬로 만들기 (padding with large values)
        n_gt, n_pred = cost_matrix.shape
        size = max(n_gt, n_pred)
        padded_matrix = np.full((size, size), 1e6)
        padded_matrix[:n_gt, :n_pred] = cost_matrix
        
        # 간단한 greedy matching (Hungarian의 근사)
        # 실제 Hungarian algorithm은 복잡하므로 개선된 greedy 사용
        matches = []
        used_gt = set()
        used_pred = set()
        
        # 모든 가능한 쌍을 거리순으로 정렬
        pairs = []
        for i in range(n_gt):
            for j in range(n_pred):
                if cost_matrix[i, j] < max_distance:
                    pairs.append((cost_matrix[i, j], i, j))
        
        # 거리순으로 정렬
        pairs.sort()
        
        # Greedy 매칭 (하지만 모든 쌍을 고려)
        for distance, gt_idx, pred_idx in pairs:
            if gt_idx not in used_gt and pred_idx not in used_pred:
                matches.append((gt_idx, pred_idx, distance))
                used_gt.add(gt_idx)
                used_pred.add(pred_idx)
        
        return matches

    @torch.no_grad() 
    def _log_aligned_gt_prediction_pairs(self, output, batch, epoch, batch_idx=0):
        """
        GT-Prediction 매칭 로깅 비활성화 (성능 최적화)
        """
        return  # 비활성화
            
        # 실제 학습에 사용되는 매칭된 데이터 (image 좌표계)
        if 'poly_init' in output:
            # network에서 매칭된 initial contour 사용 (학습에 실제 사용되는 데이터)
            aligned_pred = output['poly_init']  # (N, V, 2) - 매칭된 initial contour
        else:
            # fallback: GCN 결과 사용
            aligned_pred = output['py_pred'][1] if len(output['py_pred']) > 1 else output['py_pred'][0]
            
        aligned_gt = output['img_gt_polys']   # (N, V, 2) - 매칭된 GT
        
        print(f"\n[ALIGNED LOG] Epoch {epoch}, Batch {batch_idx}")
        print(f"  Aligned Pred shape: {aligned_pred.shape}")
        print(f"  Aligned GT shape: {aligned_gt.shape}")
        
        if aligned_pred.shape[0] == 0:
            print("  No aligned pairs to analyze")
            return
            
        # 몇 개 샘플에 대해 정렬 상태 확인
        num_check = min(3, aligned_pred.shape[0])
        for i in range(num_check):
            pred_center = aligned_pred[i].mean(dim=0)
            gt_center = aligned_gt[i].mean(dim=0)
            center_dist = torch.norm(pred_center - gt_center).item()
            
            # 첫 번째 vertex 거리
            v0_dist = torch.norm(aligned_pred[i, 0] - aligned_gt[i, 0]).item()
            
            print(f"    Pair {i}: Center dist={center_dist:.2f}, V0 dist={v0_dist:.2f}")
            
        # 시각화 저장 (실제 학습 사용 데이터 그대로)
        if epoch == 0:
            self._save_actual_training_pairs(aligned_pred, aligned_gt, epoch, batch_idx)

    @torch.no_grad()
    def _save_actual_training_pairs(self, aligned_pred, aligned_gt, epoch, batch_idx):
        """실제 학습에 사용되는 GT-Prediction 쌍들을 순서대로 그대로 시각화 (추가 처리 없음)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # 로그 디렉토리 설정
            log_dir = os.path.join(self.cfg.commen.result_dir, "actual_training_pairs")
            os.makedirs(log_dir, exist_ok=True)
            
            # CPU로 이동
            pred_np = aligned_pred.detach().cpu().numpy()
            gt_np = aligned_gt.detach().cpu().numpy()
            
            # 시각화할 쌍의 수 제한 (최대 6개)
            num_pairs = min(6, len(pred_np))
            
            if num_pairs == 0:
                print("  No pairs to visualize")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # 각 쌍을 순서대로 시각화 (i번째 pred와 i번째 gt는 매칭된 쌍)
            for i in range(num_pairs):
                ax = axes[i]
                
                # GT polygon (빨간색 실선)
                gt_polygon = patches.Polygon(gt_np[i], fill=False, edgecolor='red', linewidth=2, linestyle='-')
                ax.add_patch(gt_polygon)
                
                # Prediction polygon (파란색 점선)  
                pred_polygon = patches.Polygon(pred_np[i], fill=False, edgecolor='blue', linewidth=2, linestyle='--')
                ax.add_patch(pred_polygon)
                
                # Center 계산 및 표시
                gt_center = gt_np[i].mean(axis=0)
                pred_center = pred_np[i].mean(axis=0)
                center_dist = np.linalg.norm(gt_center - pred_center)
                
                ax.plot(gt_center[0], gt_center[1], 'ro', markersize=8, label='GT Center')
                ax.plot(pred_center[0], pred_center[1], 'bs', markersize=8, label='Pred Center')
                
                # Center 연결선
                ax.plot([gt_center[0], pred_center[0]], [gt_center[1], pred_center[1]], 
                       'g--', alpha=0.7, linewidth=2, label='Center Connection')
                
                # Vertex 매칭 시각화 (처음 8개 vertex)
                num_vertices = min(8, len(gt_np[i]))
                vertex_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'olive']
                
                for v_idx in range(num_vertices):
                    color = vertex_colors[v_idx % len(vertex_colors)]
                    
                    # GT vertex (원형)
                    ax.plot(gt_np[i, v_idx, 0], gt_np[i, v_idx, 1], 'o', color=color, markersize=4)
                    
                    # Prediction vertex (사각형)
                    ax.plot(pred_np[i, v_idx, 0], pred_np[i, v_idx, 1], 's', color=color, markersize=4)
                    
                    # 매칭된 vertex들 연결선 (점선)
                    ax.plot([gt_np[i, v_idx, 0], pred_np[i, v_idx, 0]], 
                           [gt_np[i, v_idx, 1], pred_np[i, v_idx, 1]], 
                           color=color, linestyle=':', alpha=0.6, linewidth=1)
                
                # V0 거리 계산
                v0_dist = np.linalg.norm(gt_np[i, 0] - pred_np[i, 0])
                
                ax.set_xlim(0, 416)
                ax.set_ylim(0, 416) 
                ax.invert_yaxis()
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Training Pair {i}\nCenter: {center_dist:.1f}px, V0: {v0_dist:.1f}px')
                
                if i == 0:  # 첫 번째에만 legend 표시
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='red', linewidth=2, linestyle='-', label='GT Polygon'),
                        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Pred Polygon'),
                        Line2D([0], [0], marker='o', color='gray', linewidth=0, markersize=4, label='GT Vertex'),
                        Line2D([0], [0], marker='s', color='gray', linewidth=0, markersize=4, label='Pred Vertex'),
                        Line2D([0], [0], color='gray', linewidth=1, linestyle=':', label='Vertex Match'),
                        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Center Match')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
            
            # 빈 subplot 숨기기
            for i in range(num_pairs, 6):
                axes[i].set_visible(False)
                
            plt.suptitle(f'Actual Training GT-Prediction Pairs (Epoch {epoch}, Batch {batch_idx})', fontsize=14)
            plt.tight_layout()
            
            # 저장
            filename = f"training_pairs_epoch{epoch:03d}_batch{batch_idx:03d}.png"
            filepath = os.path.join(log_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            # print(f"  [TRAINING PAIRS] Saved actual training pairs: {filepath}")
                
        except Exception as e:
            print(f"  Warning: Could not save training pairs visualization: {e}")

    @torch.no_grad()
    def _save_aligned_visualization_by_image_backup(self, aligned_pred, aligned_gt, batch_ind, epoch, batch_idx):
        """실제 학습에 사용되는 매칭된 GT-Prediction 쌍들을 이미지별로 시각화"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            if batch_ind is None:
                print("  Warning: No batch_ind found, skipping image-wise visualization")
                return
                
            # 로그 디렉토리 설정
            log_dir = os.path.join(self.cfg.commen.result_dir, "matched_gt_pred")
            os.makedirs(log_dir, exist_ok=True)
            
            # CPU로 이동
            pred_np = aligned_pred.detach().cpu().numpy()
            gt_np = aligned_gt.detach().cpu().numpy()
            batch_ind_np = batch_ind.detach().cpu().numpy()
            
            # 이미지별로 그룹화
            unique_imgs = np.unique(batch_ind_np)
            
            # 최대 4개 이미지만 시각화
            for img_idx in unique_imgs[:4]:
                mask = (batch_ind_np == img_idx)
                if not mask.any():
                    continue
                    
                img_pred = pred_np[mask]
                img_gt = gt_np[mask]
                
                if len(img_pred) == 0:
                    continue
                
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                
                # 매칭된 쌍들 그리기 (같은 색상으로)
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'olive']
                
                for pair_idx in range(min(len(img_pred), 8)):  # 최대 8개 쌍
                    color = colors[pair_idx % len(colors)]
                    
                    # GT polygon (실선)
                    gt_polygon = patches.Polygon(img_gt[pair_idx], fill=False, edgecolor=color, 
                                               linewidth=2, linestyle='-')
                    ax.add_patch(gt_polygon)
                    
                    # Prediction polygon (점선)  
                    pred_polygon = patches.Polygon(img_pred[pair_idx], fill=False, edgecolor=color, 
                                                 linewidth=2, linestyle='--')
                    ax.add_patch(pred_polygon)
                    
                    # Center 표시
                    gt_center = img_gt[pair_idx].mean(axis=0)
                    pred_center = img_pred[pair_idx].mean(axis=0)
                    
                    ax.plot(gt_center[0], gt_center[1], 'o', color=color, markersize=8)
                    ax.plot(pred_center[0], pred_center[1], 's', color=color, markersize=8)
                    
                    # Center 연결선
                    ax.plot([gt_center[0], pred_center[0]], [gt_center[1], pred_center[1]], 
                           color=color, linestyle=':', alpha=0.7, linewidth=2)
                    
                    # Center distance 표시
                    center_dist = np.linalg.norm(gt_center - pred_center)
                    mid_x = (gt_center[0] + pred_center[0]) / 2
                    mid_y = (gt_center[1] + pred_center[1]) / 2
                    ax.text(mid_x, mid_y, f'{center_dist:.1f}', color=color, fontsize=9, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                ax.set_xlim(0, 416)
                ax.set_ylim(0, 416)
                ax.invert_yaxis()
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Matched GT-Pred Pairs (Epoch {epoch}, Batch {batch_idx}, Image {img_idx})\n'
                           f'Pairs: {len(img_pred)}')
                
                # Legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='GT'),
                    Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Pred (Matched)'),
                    Line2D([0], [0], color='black', linewidth=2, linestyle=':', label='Center Match')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
                # 저장
                filename = f"matched_pairs_epoch{epoch:03d}_batch{batch_idx:03d}_img{img_idx:02d}.png"
                filepath = os.path.join(log_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
            print(f"  [MATCHED LOG] Saved image-wise visualizations to: {log_dir}")
                
        except Exception as e:
            print(f"  Warning: Could not save image-wise visualization: {e}")

    @torch.no_grad()
    def _save_aligned_visualization_backup(self, aligned_pred, aligned_gt, epoch, batch_idx):
        """이미 정렬된 GT-Prediction 쌍들의 시각화 저장"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # 로그 디렉토리 설정
            log_dir = os.path.join(self.cfg.commen.result_dir, "aligned_gt_pred")
            os.makedirs(log_dir, exist_ok=True)
            
            # CPU로 이동
            pred_np = aligned_pred.detach().cpu().numpy()
            gt_np = aligned_gt.detach().cpu().numpy()
            
            # 시각화할 쌍의 수 제한
            num_pairs = min(6, pred_np.shape[0])
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i in range(num_pairs):
                ax = axes[i]
                
                # GT polygon (빨간색 실선)
                gt_polygon = patches.Polygon(gt_np[i], fill=False, edgecolor='red', linewidth=2, linestyle='-')
                ax.add_patch(gt_polygon)
                
                # Prediction polygon (파란색 점선)  
                pred_polygon = patches.Polygon(pred_np[i], fill=False, edgecolor='blue', linewidth=2, linestyle='--')
                ax.add_patch(pred_polygon)
                
                # vertex 매칭 상태 시각화 (모든 vertex 또는 최대 8개)
                num_vertices = min(8, pred_np.shape[1])
                vertex_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'olive']
                
                for v_idx in range(num_vertices):
                    color = vertex_colors[v_idx % len(vertex_colors)]
                    
                    # GT vertex (원형)
                    ax.plot(gt_np[i, v_idx, 0], gt_np[i, v_idx, 1], 'o', color=color, markersize=6)
                    ax.text(gt_np[i, v_idx, 0]+3, gt_np[i, v_idx, 1]+3, f'G{v_idx}', color=color, fontsize=8, weight='bold')
                    
                    # Prediction vertex (사각형)
                    ax.plot(pred_np[i, v_idx, 0], pred_np[i, v_idx, 1], 's', color=color, markersize=6)
                    ax.text(pred_np[i, v_idx, 0]+3, pred_np[i, v_idx, 1]+3, f'P{v_idx}', color=color, fontsize=8, weight='bold')
                    
                    # 매칭된 vertex들 연결선 (점선)
                    ax.plot([gt_np[i, v_idx, 0], pred_np[i, v_idx, 0]], 
                           [gt_np[i, v_idx, 1], pred_np[i, v_idx, 1]], 
                           color=color, linestyle=':', alpha=0.7, linewidth=1.5)
                    
                    # vertex 간 거리 표시 (중간점에)
                    if v_idx < 4:  # 처음 4개 vertex만 거리 표시 (너무 많으면 복잡)
                        mid_x = (gt_np[i, v_idx, 0] + pred_np[i, v_idx, 0]) / 2
                        mid_y = (gt_np[i, v_idx, 1] + pred_np[i, v_idx, 1]) / 2
                        v_dist = np.linalg.norm(gt_np[i, v_idx] - pred_np[i, v_idx])
                        ax.text(mid_x, mid_y, f'{v_dist:.1f}', color=color, fontsize=7, weight='bold',
                               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
                
                center_dist = np.linalg.norm(pred_np[i].mean(axis=0) - gt_np[i].mean(axis=0))
                ax.set_title(f'Pair {i} (center_dist={center_dist:.1f})')
                ax.set_xlim(0, 416)
                ax.set_ylim(0, 416) 
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)
                
                # Legend 업데이트 (vertex 매칭 시각화 설명)
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='red', linewidth=2, linestyle='-', label='GT Polygon'),
                    Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Pred Polygon'),
                    Line2D([0], [0], marker='o', color='gray', linewidth=0, markersize=6, label='GT Vertex'),
                    Line2D([0], [0], marker='s', color='gray', linewidth=0, markersize=6, label='Pred Vertex'),
                    Line2D([0], [0], color='gray', linewidth=1.5, linestyle=':', label='Vertex Match')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            # 빈 subplot 숨기기
            for i in range(num_pairs, 6):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            
            # 저장
            filename = f"aligned_pairs_epoch{epoch:03d}_batch{batch_idx:03d}.png"
            filepath = os.path.join(log_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  [ALIGNED LOG] Saved visualization: {filepath}")
            
        except Exception as e:
            print(f"  Warning: Could not save aligned visualization: {e}")

    @torch.no_grad()
    def _log_gt_prediction_matching_original_backup(self, output, batch, epoch, batch_idx=0):
        """
        원본 batch 데이터를 직접 사용한 GT-Prediction 매칭 로깅
        순서 불일치 문제 해결 - gradient 계산에 영향 없음
        """
        import torch
        import numpy as np
        import os
        
        # 원본 GT 데이터 (feature map 좌표계)
        original_gt_polys = batch['img_gt_polys']  # (B, max_len, V, 2)
        ct_01 = batch['ct_01']  # (B, max_len) - valid GT mask
        
        # Prediction 데이터 
        if 'poly_init' not in output or 'batch_ind' not in output:
            print(f"[MATCH LOG] Epoch {epoch}, Batch {batch_idx}: No prediction data")
            return
            
        pred_polys = output['poly_init']  # (N, V, 2) - image 좌표계
        batch_ind = output['batch_ind']   # (N,) - 각 prediction이 속한 이미지 인덱스
        
        print(f"\n[ORIGINAL MATCH LOG] Epoch {epoch}, Batch {batch_idx}")
        print(f"  Original GT shape: {original_gt_polys.shape}, ct_01 shape: {ct_01.shape}")
        print(f"  Prediction shape: {pred_polys.shape}, batch_ind shape: {batch_ind.shape}")
        
        # 좌표계 통일: prediction을 feature map 좌표계로 변환
        pred_polys_feature = pred_polys / self.cfg.commen.down_ratio  # image → feature
        
        B, max_len, V, _ = original_gt_polys.shape
        device = original_gt_polys.device
        
        # 이미지별로 매칭 수행
        for img_idx in range(B):
            # 해당 이미지의 유효한 GT들
            valid_mask = ct_01[img_idx]  # (max_len,)
            if not valid_mask.any():
                continue
                
            valid_gt_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            gt_polys_img = original_gt_polys[img_idx][valid_gt_indices]  # (num_valid_gt, V, 2)
            
            # 해당 이미지의 Prediction들
            pred_mask = (batch_ind == img_idx)
            if not pred_mask.any():
                continue
                
            pred_polys_img = pred_polys_feature[pred_mask]  # (num_pred, V, 2)
            
            print(f"    Image {img_idx}: GT={len(gt_polys_img)}, Pred={len(pred_polys_img)}")
            
            if len(gt_polys_img) == 0 or len(pred_polys_img) == 0:
                continue
                
            # Center 계산
            gt_centers = gt_polys_img.mean(dim=1)  # (num_gt, 2)
            pred_centers = pred_polys_img.mean(dim=1)  # (num_pred, 2)
            
            # 거리 계산 (gradient 연결 끊기)
            distances = torch.cdist(gt_centers.detach(), pred_centers.detach())  # (num_gt, num_pred)
            
            # Greedy 매칭 (성능 최적화)
            max_distance = getattr(self.cfg.train, 'max_center_distance_maskinit', 50.0)
            matches = self._greedy_matching(distances.detach().cpu().numpy(), max_distance)
            
            print(f"      Matches: {len(matches)}/{min(len(gt_polys_img), len(pred_polys_img))}")
            
            # 매칭 결과 출력 (첫 1개만, 성능 최적화)
            if len(matches) > 0:
                gt_idx, pred_idx, distance = matches[0]
                print(f"        GT{valid_gt_indices[gt_idx].item()}<->Pred{torch.nonzero(pred_mask)[pred_idx].item()}: {distance:.2f}")
            
            # 시각화 저장 (첫 4개 이미지, epoch 0만)
            if len(matches) > 0 and img_idx < 4 and epoch == 0:
                # 좌표계를 image 좌표계로 변환하여 시각화 (CPU로 이동)
                gt_polys_img_vis = (gt_polys_img * self.cfg.commen.down_ratio).cpu()  # feature → image
                pred_polys_img_vis = (pred_polys_img * self.cfg.commen.down_ratio).cpu()  # feature → image
                gt_centers_vis = (gt_centers * self.cfg.commen.down_ratio).cpu()
                pred_centers_vis = (pred_centers * self.cfg.commen.down_ratio).cpu()
                
                # 로그 디렉토리 설정
                import os
                log_dir = os.path.join(self.cfg.commen.result_dir, "gt_pred_matching")
                os.makedirs(log_dir, exist_ok=True)
                
                self._save_matching_visualization_by_image(
                    gt_polys_img_vis, pred_polys_img_vis, gt_centers_vis, pred_centers_vis,
                    epoch, batch_idx, log_dir, matches, batch_ind, batch, img_idx
                )
                
                # Vertex 매칭 시각화도 저장
                self._save_vertex_matching_visualization(
                    gt_polys_img_vis, pred_polys_img_vis, matches,
                    epoch, batch_idx, log_dir, img_idx
                )

    @torch.no_grad()
    def _log_gt_prediction_matching_py_pred(self, output, batch, py_pred, epoch, batch_idx=0):
        """
        py_pred[1]과 GT 매칭 로깅 (GCN 거친 결과 확인)
        gradient 계산에 영향 없음 - 로깅/시각화 전용
        """
        import torch
        import numpy as np
        import os
        
        # 원본 GT 데이터 (feature map 좌표계)
        original_gt_polys = batch['img_gt_polys']  # (B, max_len, V, 2)
        ct_01 = batch['ct_01']  # (B, max_len) - valid GT mask
        
        # py_pred[1] 사용 (GCN 거친 결과)
        if len(py_pred) <= 1:
            print(f"[PY_PRED MATCH LOG] Epoch {epoch}, Batch {batch_idx}: py_pred[1] not available")
            return
            
        pred_polys = py_pred[1]  # (N, V, 2) - image 좌표계
        
        # batch_ind 가져오기
        if 'batch_ind' not in output:
            print(f"[PY_PRED MATCH LOG] Epoch {epoch}, Batch {batch_idx}: No batch_ind")
            return
        batch_ind = output['batch_ind']   # (N,) - 각 prediction이 속한 이미지 인덱스
        
        print(f"\n[PY_PRED MATCH LOG] Epoch {epoch}, Batch {batch_idx}")
        print(f"  Original GT shape: {original_gt_polys.shape}, ct_01 shape: {ct_01.shape}")
        print(f"  py_pred[1] shape: {pred_polys.shape}, batch_ind shape: {batch_ind.shape}")
        
        # 좌표계 통일: prediction을 feature map 좌표계로 변환
        pred_polys_feature = pred_polys / self.cfg.commen.down_ratio  # image → feature
        
        B, max_len, V, _ = original_gt_polys.shape
        device = original_gt_polys.device
        
        # 이미지별로 매칭 수행 (첫 4개 이미지만)
        for img_idx in range(min(4, B)):
            # 해당 이미지의 유효한 GT들
            valid_mask = ct_01[img_idx]  # (max_len,)
            if not valid_mask.any():
                continue
                
            valid_gt_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            gt_polys_img = original_gt_polys[img_idx][valid_gt_indices]  # (num_valid_gt, V, 2)
            
            # 해당 이미지의 Prediction들
            pred_mask = (batch_ind == img_idx)
            if not pred_mask.any():
                continue
                
            pred_polys_img = pred_polys_feature[pred_mask]  # (num_pred, V, 2)
            
            print(f"    Image {img_idx}: GT={len(gt_polys_img)}, py_pred[1]={len(pred_polys_img)}")
            
            if len(gt_polys_img) == 0 or len(pred_polys_img) == 0:
                continue
                
            # Center 계산
            gt_centers = gt_polys_img.mean(dim=1)  # (num_gt, 2)
            pred_centers = pred_polys_img.mean(dim=1)  # (num_pred, 2)
            
            # 거리 계산 (gradient 연결 끊기)
            distances = torch.cdist(gt_centers.detach(), pred_centers.detach())  # (num_gt, num_pred)
            
            # Greedy 매칭
            max_distance = getattr(self.cfg.train, 'max_center_distance_maskinit', 50.0)
            matches = self._greedy_matching(distances.detach().cpu().numpy(), max_distance)
            
            print(f"      Matches: {len(matches)}/{min(len(gt_polys_img), len(pred_polys_img))}")
            
            # 매칭 결과 출력 (첫 1개만)
            if len(matches) > 0:
                gt_idx, pred_idx, distance = matches[0]
                print(f"        GT{valid_gt_indices[gt_idx].item()}<->py_pred[1]_{torch.nonzero(pred_mask)[pred_idx].item()}: {distance:.2f}")
            
            # 시각화 저장
            if len(matches) > 0:
                # 좌표계를 image 좌표계로 변환하여 시각화 (gradient 연결 끊고 CPU로 이동)
                gt_polys_img_vis = (gt_polys_img * self.cfg.commen.down_ratio).detach().cpu()  # feature → image
                pred_polys_img_vis = (pred_polys_img * self.cfg.commen.down_ratio).detach().cpu()  # feature → image
                gt_centers_vis = (gt_centers * self.cfg.commen.down_ratio).detach().cpu()
                pred_centers_vis = (pred_centers * self.cfg.commen.down_ratio).detach().cpu()
                
                # 로그 디렉토리 설정
                log_dir = os.path.join(self.cfg.commen.result_dir, "gt_py_pred_matching")
                os.makedirs(log_dir, exist_ok=True)
                
                # 기존 함수 재사용 (파일명만 다르게)
                self._save_py_pred_visualization(
                    gt_polys_img_vis, pred_polys_img_vis, gt_centers_vis, pred_centers_vis,
                    epoch, batch_idx, log_dir, matches, img_idx
                )
                
                # py_pred[1] Vertex 매칭 시각화도 저장
                self._save_py_pred_vertex_visualization(
                    gt_polys_img_vis, pred_polys_img_vis, matches,
                    epoch, batch_idx, log_dir, img_idx
                )
                
                # py_pred[0] vs py_pred[1] 비교 (변화량 확인)
                if len(py_pred) > 1:
                    self._save_evolution_comparison(
                        py_pred[0][pred_mask], py_pred[1][pred_mask], 
                        epoch, batch_idx, log_dir, img_idx
                    )

    @torch.no_grad()
    def _save_py_pred_visualization(self, gt_polys_img, pred_polys_img, gt_centers, pred_centers, 
                                   epoch, batch_idx, log_dir, matches, img_idx=0):
        """py_pred[1] 매칭 시각화 저장 - gradient 계산에 영향 없음"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
            
            # 간단하게 매칭 결과 저장
            gt_polys_np = gt_polys_img.detach().cpu().numpy()
            pred_polys_np = pred_polys_img.detach().cpu().numpy()
            gt_centers_np = gt_centers.detach().cpu().numpy() 
            pred_centers_np = pred_centers.detach().cpu().numpy()
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # 매칭된 쌍들 그리기 (같은 색상으로)
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan', 'magenta']
            
            print(f"  [DEBUG] Saving py_pred[1] visualization with {len(matches)} matches")
            
            for match_idx, (gt_idx, pred_idx, distance) in enumerate(matches[:5]):  # 최대 5개만
                color = colors[match_idx % len(colors)]
                
                # GT polygon (실선)
                if gt_idx < len(gt_polys_np):
                    polygon = patches.Polygon(gt_polys_np[gt_idx], fill=False, edgecolor=color, linewidth=2, 
                                            linestyle='-', label=f'GT' if match_idx == 0 else "")
                    ax.add_patch(polygon)
                    ax.plot(gt_centers_np[gt_idx][0], gt_centers_np[gt_idx][1], 'o', color=color, markersize=8)
                    ax.text(gt_centers_np[gt_idx][0]+3, gt_centers_np[gt_idx][1]+3, f'GT{gt_idx}', color=color, fontsize=10, weight='bold')
                
                # py_pred[1] polygon (점선, 같은 색상)
                if pred_idx < len(pred_polys_np):
                    polygon = patches.Polygon(pred_polys_np[pred_idx], fill=False, edgecolor=color, linewidth=2,
                                            linestyle='--', label=f'py_pred[1]' if match_idx == 0 else "")
                    ax.add_patch(polygon)
                    ax.plot(pred_centers_np[pred_idx][0], pred_centers_np[pred_idx][1], 's', color=color, markersize=8)
                    ax.text(pred_centers_np[pred_idx][0]+3, pred_centers_np[pred_idx][1]+3, f'PP{pred_idx}', color=color, fontsize=10, weight='bold')
                    
                    # 매칭 라인 그리기
                    ax.plot([gt_centers_np[gt_idx][0], pred_centers_np[pred_idx][0]], 
                           [gt_centers_np[gt_idx][1], pred_centers_np[pred_idx][1]], 
                           color=color, linestyle=':', alpha=0.8, linewidth=2)
                    
                    # 거리 텍스트 표시
                    mid_x = (gt_centers_np[gt_idx][0] + pred_centers_np[pred_idx][0]) / 2
                    mid_y = (gt_centers_np[gt_idx][1] + pred_centers_np[pred_idx][1]) / 2
                    ax.text(mid_x, mid_y, f'{distance:.1f}', color=color, fontsize=9, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            ax.set_xlim(0, 416)
            ax.set_ylim(0, 416)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'GT vs py_pred[1] Matching (Epoch {epoch}, Batch {batch_idx}, Image {img_idx})\n'
                       f'Matches: {len(matches)}, GT: {len(gt_polys_np)}, py_pred[1]: {len(pred_polys_np)}')
            ax.legend(loc='upper right')
            
            # 저장
            img_filename = f"py_pred_matching_epoch{epoch:03d}_batch{batch_idx:03d}_img{img_idx:02d}.png"
            img_path = os.path.join(log_dir, img_filename)
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  [DEBUG] Saved py_pred[1] visualization to: {img_path}")
                
        except Exception as e:
            print(f"  Warning: Could not save py_pred[1] visualization: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def _save_py_pred_vertex_visualization(self, gt_polys_img, pred_polys_img, matches, 
                                          epoch, batch_idx, log_dir, img_idx=0):
        """py_pred[1]의 vertex별 매칭 상황 시각화"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
            
            gt_polys_np = gt_polys_img.detach().cpu().numpy()
            pred_polys_np = pred_polys_img.detach().cpu().numpy()
            
            # 매칭된 첫 3개 쌍에 대해 vertex 시각화
            num_visualize = min(3, len(matches))
            if num_visualize == 0:
                return
            
            fig, axes = plt.subplots(1, num_visualize, figsize=(6*num_visualize, 6))
            if num_visualize == 1:
                axes = [axes]
            
            colors = ['red', 'blue', 'green']
            
            for match_idx in range(num_visualize):
                gt_idx, pred_idx, distance = matches[match_idx]
                ax = axes[match_idx]
                color = colors[match_idx]
                
                if gt_idx >= len(gt_polys_np) or pred_idx >= len(pred_polys_np):
                    continue
                
                gt_poly = gt_polys_np[gt_idx]
                pred_poly_aligned = pred_polys_np[pred_idx]  # 이미 poly_init에서 정렬됨
                
                # GT contour 그리기 (실선)
                polygon = patches.Polygon(gt_poly, fill=False, edgecolor=color, 
                                        linewidth=2, linestyle='-', alpha=0.8, label='GT')
                ax.add_patch(polygon)
                
                # py_pred[1] contour 그리기 (점선) - 정렬된 버전
                polygon = patches.Polygon(pred_poly_aligned, fill=False, edgecolor=color,
                                        linewidth=2, linestyle='--', alpha=0.8, label='py_pred[1]')
                ax.add_patch(polygon)
                
                # Vertex 번호 표시 및 매칭 라인
                num_vertices = min(len(gt_poly), len(pred_poly_aligned))
                for v_idx in range(num_vertices):
                    gt_v = gt_poly[v_idx]
                    pred_v = pred_poly_aligned[v_idx]
                    
                    # GT vertex (원형 마커)
                    ax.plot(gt_v[0], gt_v[1], 'o', color=color, markersize=8, 
                           markeredgecolor='white', markeredgewidth=1)
                    ax.text(gt_v[0]+2, gt_v[1]+2, f'G{v_idx}', color=color, 
                           fontsize=8, weight='bold')
                    
                    # py_pred[1] vertex (사각형 마커)
                    ax.plot(pred_v[0], pred_v[1], 's', color=color, markersize=8,
                           markeredgecolor='white', markeredgewidth=1)
                    ax.text(pred_v[0]+2, pred_v[1]+2, f'P{v_idx}', color=color,
                           fontsize=8, weight='bold')
                    
                    # Vertex 매칭 라인
                    ax.plot([gt_v[0], pred_v[0]], [gt_v[1], pred_v[1]], 
                           color=color, linestyle=':', alpha=0.6, linewidth=1)
                    
                    # Vertex 거리 표시
                    v_dist = np.linalg.norm(gt_v - pred_v)
                    mid_x = (gt_v[0] + pred_v[0]) / 2
                    mid_y = (gt_v[1] + pred_v[1]) / 2
                    ax.text(mid_x, mid_y, f'{v_dist:.1f}', color=color, fontsize=7,
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
                
                # 축 설정
                ax.set_xlim(min(gt_poly[:, 0].min(), pred_poly_aligned[:, 0].min()) - 20,
                           max(gt_poly[:, 0].max(), pred_poly_aligned[:, 0].max()) + 20)
                ax.set_ylim(min(gt_poly[:, 1].min(), pred_poly_aligned[:, 1].min()) - 20,
                           max(gt_poly[:, 1].max(), pred_poly_aligned[:, 1].max()) + 20)
                ax.invert_yaxis()
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f'GT{gt_idx} ↔ py_pred[1]{pred_idx} (Vertex Aligned)\nCenter Dist: {distance:.2f}px')
                ax.legend()
            
            plt.suptitle(f'py_pred[1] Vertex Matching (Epoch {epoch}, Batch {batch_idx}, Image {img_idx})')
            plt.tight_layout()
            
            # 저장
            vertex_filename = f"py_pred_vertex_matching_epoch{epoch:03d}_batch{batch_idx:03d}_img{img_idx:02d}.png"
            vertex_path = os.path.join(log_dir, vertex_filename)
            plt.savefig(vertex_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  [DEBUG] Saved py_pred[1] vertex visualization to: {vertex_path}")
            
        except Exception as e:
            print(f"  Warning: Could not save py_pred[1] vertex visualization: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def _save_evolution_comparison(self, py_pred_0, py_pred_1, epoch, batch_idx, log_dir, img_idx=0):
        """py_pred[0] vs py_pred[1] 변화량 시각화 (GCN evolution 확인)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
            
            # 좌표계 변환 (image 좌표계로)
            pred_0_vis = (py_pred_0 * self.cfg.commen.down_ratio).detach().cpu().numpy()
            pred_1_vis = (py_pred_1 * self.cfg.commen.down_ratio).detach().cpu().numpy()
            
            # 첫 3개 contour만 비교
            num_visualize = min(3, len(pred_0_vis), len(pred_1_vis))
            if num_visualize == 0:
                return
            
            fig, axes = plt.subplots(1, num_visualize, figsize=(6*num_visualize, 6))
            if num_visualize == 1:
                axes = [axes]
            
            colors = ['red', 'blue', 'green']
            
            total_movement = 0
            for i in range(num_visualize):
                ax = axes[i]
                color = colors[i]
                
                poly_0 = pred_0_vis[i]  # py_pred[0]
                poly_1 = pred_1_vis[i]  # py_pred[1]
                
                # py_pred[0] 그리기 (실선)
                polygon = patches.Polygon(poly_0, fill=False, edgecolor=color, 
                                        linewidth=2, linestyle='-', alpha=0.8, label='py_pred[0]')
                ax.add_patch(polygon)
                
                # py_pred[1] 그리기 (점선)
                polygon = patches.Polygon(poly_1, fill=False, edgecolor='orange',
                                        linewidth=2, linestyle='--', alpha=0.8, label='py_pred[1]')
                ax.add_patch(polygon)
                
                # 변화 벡터 그리기
                vertex_movements = []
                for v_idx in range(len(poly_0)):
                    v0 = poly_0[v_idx]
                    v1 = poly_1[v_idx]
                    
                    # 변화 벡터
                    ax.arrow(v0[0], v0[1], v1[0]-v0[0], v1[1]-v0[1], 
                            head_width=3, head_length=3, fc='green', ec='green', alpha=0.7)
                    
                    # 변화량 계산
                    movement = np.linalg.norm(v1 - v0)
                    vertex_movements.append(movement)
                    
                    # Vertex 번호
                    ax.text(v0[0]+3, v0[1]+3, f'{v_idx}', color=color, fontsize=8)
                
                avg_movement = np.mean(vertex_movements)
                total_movement += avg_movement
                
                # 축 설정
                all_points = np.concatenate([poly_0, poly_1])
                ax.set_xlim(all_points[:, 0].min() - 20, all_points[:, 0].max() + 20)
                ax.set_ylim(all_points[:, 1].min() - 20, all_points[:, 1].max() + 20)
                ax.invert_yaxis()
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Contour {i} Evolution\nAvg Movement: {avg_movement:.2f}px')
                ax.legend()
            
            avg_total_movement = total_movement / num_visualize
            plt.suptitle(f'py_pred[0] → py_pred[1] Evolution (Epoch {epoch}, Batch {batch_idx}, Image {img_idx})\n'
                        f'Average Movement: {avg_total_movement:.2f}px')
            plt.tight_layout()
            
            # 저장
            evolution_filename = f"evolution_comparison_epoch{epoch:03d}_batch{batch_idx:03d}_img{img_idx:02d}.png"
            evolution_path = os.path.join(log_dir, evolution_filename)
            plt.savefig(evolution_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  [DEBUG] Saved evolution comparison to: {evolution_path}")
            print(f"  [DEBUG] Average vertex movement: {avg_total_movement:.2f}px")
            
        except Exception as e:
            print(f"  Warning: Could not save evolution comparison: {e}")
            import traceback
            traceback.print_exc()

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
