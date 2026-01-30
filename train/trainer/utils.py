import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import warnings
import random
from scipy.optimize import linear_sum_assignment  # 헝가리안 알고리즘

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) # (batch, n_obj) -> (batch, n_obj, 1) -> (batch, n_obj, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _downsample_closed_poly(gt_hi: torch.Tensor, target_v: int, fallback_fn=None) -> torch.Tensor:
    """
    gt_hi: (B, Vmax, 2)  # 최대 해상도 GT (예: 96)
    target_v: 48, 96 등
    fallback_fn: callable(poly, target_v) -> (B,target_v,2)  # 안 나눠떨어질 때 리샘플 함수

    return: (B, target_v, 2)
    """
    if gt_hi.numel() == 0:
        return gt_hi
    B, Vmax, _ = gt_hi.shape
    if Vmax % target_v == 0:
        stride = Vmax // target_v
        return gt_hi[:, ::stride, :]
    # 안전 폴백 (예: Vmax=96, target_v=64 같은 특수 케이스)
    if fallback_fn is not None:
        return fallback_fn(gt_hi, target_v)
    # 마지막 폴백: 최근접 인덱스 매핑
    idx = torch.linspace(0, Vmax-1, steps=target_v, device=gt_hi.device).round().long()
    return gt_hi.index_select(1, idx)

def _resample_closed_poly_batch(poly: torch.Tensor, target_v: int) -> torch.Tensor:
    """
    poly: (B, V, 2)  — 닫힌 폴리곤(마지막-첫점 연결 가정)
    return: (B, target_v, 2)
    """
    if poly.numel() == 0:
        return poly

    B, V, _ = poly.shape
    pts = poly
    nxt = torch.roll(pts, shifts=-1, dims=1)
    seg = nxt - pts                       # (B,V,2)
    seg_len = torch.linalg.norm(seg, dim=-1) + 1e-8  # (B,V)
    perim = seg_len.sum(dim=1, keepdim=True)         # (B,1)
    t = torch.cumsum(seg_len, dim=1) / perim         # (B,V) in (0,1], cumulative
    t0 = torch.zeros((B,1), device=poly.device, dtype=poly.dtype)
    tau = torch.cat([t0, t[:, :-1]], dim=1)          # (B,V) start param each edge

    # target parameters: 0..1  (exclude 1.0 to avoid duplicate of 0.0)
    u = torch.linspace(0, 1, steps=target_v+1, device=poly.device, dtype=poly.dtype)[:-1]  # (target_v,)
    u = u.unsqueeze(0).expand(B, -1)  # (B,target_v)

    # 각 u가 어떤 edge에 속하는지 찾기
    # tau[b]: V개 시작점, 다음 시작점으로 넘어가기 전까지 그 edge
    # broadcast 비교: (B,target_v, V)
    mask = (u.unsqueeze(-1) >= tau.unsqueeze(1)) & (u.unsqueeze(-1) < t.unsqueeze(1))
    # 마지막 u == 0 (정확히 시작점) 같은 경계 케이스 보정
    # 그래도 모든 u는 하나의 엣지에만 매핑되도록 argmax로 인덱스 얻기
    edge_idx = mask.float().argmax(dim=-1)  # (B,target_v) in [0..V-1]

    # 해당 edge의 시작점과 벡터, edge 길이, 그리고 edge 내 비율 s 계산
    tau_k = torch.gather(tau, 1, edge_idx)                   # (B,target_v)
    seg_len_k = torch.gather(seg_len, 1, edge_idx)           # (B,target_v)
    s = (u - tau_k) * perim  / seg_len_k                     # (B,target_v) in [0,1)

    idx0 = edge_idx                                          # (B,target_v)
    p0 = torch.gather(pts, 1, idx0.unsqueeze(-1).expand(-1,-1,2))     # (B,target_v,2)
    v  = torch.gather(seg, 1, idx0.unsqueeze(-1).expand(-1,-1,2))     # (B,target_v,2)

    resampled = p0 + s.unsqueeze(-1) * v                     # (B,target_v,2)
    return resampled



class VertexClsLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', rho=5., is_focal=False, **kwargs):
        super().__init__()
        self.weight = weight if weight is None else torch.tensor(weight, dtype=torch.float32)
        self.reduction = reduction
        self.rho = rho
        self.is_focal = is_focal
        self.gamma = kwargs.get('gamma', 1.)

    def forward(self, pred_vertex_logits, pred_coords, vertex_gt_coord, keyPointsMask):
        """
        "pred_vertex_logits": B x N x C  (N: max vertices, C: num_classes)
        "vertex_gt_coord": list of tensors [num_gt_vertices_i x 2] (x, y coords)
        "pred_vertex_coord": B x N x 2
        """
        num_contours = pred_vertex_logits.shape[0]
        confidence = F.softmax(pred_vertex_logits, dim=-1)  # (B, N, C)
        # print(f"confidence : {confidence.shape}")

        total_loss = 0.0
        total_matches = 0

        for b in range(num_contours):
            pred = pred_coords[b]  # N x 2
            # print(f"vertex_gt_coord[b] : {vertex_gt_coord[b].shape}, keyPointsMask[b] : {keyPointsMask[b].shape} -> {keyPointsMask[b].sum()}")
            gt_coords = vertex_gt_coord[b][keyPointsMask[b].bool()]  # M x 2
            # print(f"gt_coords : {gt_coords.shape}")
            if gt_coords.numel() == 0:
                continue

            # Hungarian matching based on L_match
            # print(f"confidence[b,...,1] : {confidence[b,...,1].shape}")
            cost = (-confidence[b,...,1]) + self.rho * torch.cdist(gt_coords.unsqueeze(0), pred.unsqueeze(0)).squeeze(0)  # M x N
            gt_idx, pred_idx = linear_sum_assignment(cost.detach().cpu().numpy())
            # print(f"gt_idx : {gt_idx.max()} of {gt_idx.shape}, pred_idx : {pred_idx.max()} of {pred_idx.shape}")

            # matched predictions
            # print(f"pred_vertex_logits[b] : {pred_vertex_logits[b].shape}, pred_ids : {pred_idx.min()}, {pred_idx.max()}")
            matched_logits = pred_vertex_logits[b]  # N x C
            matched_labels = torch.zeros(matched_logits.size(0), dtype=torch.long,
                                        device=matched_logits.device)  # N
            matched_labels[pred_idx] = 1 #ids for pred
            weight = self.weight.to(matched_logits.device) if self.weight is not None else None

            loss = F.cross_entropy(matched_logits, matched_labels, weight=weight, reduction='none')
            if self.is_focal:
                pt = torch.exp(-loss)
                loss = (1 - pt) ** self.gamma * loss

            total_loss += loss.sum()
            total_matches += len(matched_labels)

        if total_matches == 0:
            return torch.tensor(0.0, device=pred_vertex_logits.device, requires_grad=True)
        return total_loss / total_matches

class SegmentationMetrics(object):
    r"""Calculate common metrics in semantic segmentation to evalueate model preformance.

    Supported metrics: Pixel accuracy, Dice Coeff, precision score and recall score.

    Pixel accuracy measures how many pixels in a image are predicted correctly.

    Dice Coeff is a measure function to measure similarity over 2 sets, which is usually used to
    calculate the similarity of two samples. Dice equals to f1 score in semantic segmentation tasks.

    It should be noted that Dice Coeff and Intersection over Union are highly related, so you need
    NOT calculate these metrics both, the other can be calcultaed directly when knowing one of them.

    Precision describes the purity of our positive detections relative to the ground truth. Of all
    the objects that we predicted in a given image, precision score describes how many of those objects
    actually had a matching ground truth annotation.

    Recall describes the completeness of our positive predictions relative to the ground truth. Of
    all the objected annotated in our ground truth, recall score describes how many true positive instances
    we have captured in semantic segmentation.

    Args:
        eps: float, a value added to the denominator for numerical stability.
            Default: 1e-5

        average: bool. Default: ``True``
            When set to ``True``, average Dice Coeff, precision and recall are
            returned. Otherwise Dice Coeff, precision and recall of each class
            will be returned as a numpy array.

        ignore_background: bool. Default: ``True``
            When set to ``True``, the class will not calculate related metrics on
            background pixels. When the segmentation of background pixels is not
            important, set this value to ``True``.

        activation: [None, 'none', 'softmax' (default), 'sigmoid', '0-1']
            This parameter determines what kind of activation function that will be
            applied on model output.

    Input:
        y_true: :math:`(N, H, W)`, torch tensor, where we use int value between (0, num_class - 1)
        to denote every class, where ``0`` denotes background class.
        y_pred: :math:`(N, C, H, W)`, torch tensor.

    Examples::
        >>> metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
        >>> pixel_accuracy, dice, precision, recall = metric_calculator(y_true, y_pred)
    """

    def __init__(self, eps=1e-5, average=True, ignore_background=True, activation='0-1', use_cpu=True):
        self.eps = eps
        self.average = average
        self.ignore = ignore_background
        self.activation = activation
        self.use_cpu = use_cpu
        self.stacked_true = []
        self.stacked_pred = []

    @staticmethod
    def _one_hot(gt, pred, class_num):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        input_shape = tuple(gt.shape)  # (N, H, W, ...)
        new_shape = (input_shape[0], class_num) + input_shape[1:]
        one_hot = torch.zeros(new_shape).to(pred.device, dtype=torch.float)
        target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)
        return target

    @staticmethod
    def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((3, class_num))

        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            pred_flat = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )
            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp

            matrix[0, i] = tp
            matrix[1, i] = fp
            matrix[2, i] = fn

        return matrix

    def _calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:]

        # tp = np.sum(matrix[0, :])
        # fp = np.sum(matrix[1, :])
        # fn = np.sum(matrix[2, :])

        pixel_acc = (np.sum(matrix[0, :]) + self.eps) / (np.sum(matrix[0, :]) + np.sum(matrix[1, :]))
        dice = (2 * matrix[0] + self.eps) / (2 * matrix[0] + matrix[1] + matrix[2] + self.eps)
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)

        if self.average:
            dice = np.average(dice)
            precision = np.average(precision)
            recall = np.average(recall)

        return pixel_acc, dice, precision, recall

    def stack_results(self, gt, pred):
        assert gt.shape[0] == pred.shape[0]
        assert gt.shape[-2:] == pred.shape[-2:]
        if self.use_cpu:
            self.stacked_true.append(gt.detach().cpu())
            self.stacked_pred.append(pred.detach().cpu())
        else:
            self.stacked_true.append(gt)
            self.stacked_pred.append(pred)

    def reset_results(self):
        self.stacked_true = []
        self.stacked_pred = []

    def _get_stacked_gt(self):
        return torch.cat(self.stacked_true, dim=0)

    def _get_stacked_pred(self):
        return torch.cat(self.stacked_pred, dim=0)

    def __call__(self, y_true=None, y_pred=None):
        if y_pred is None:
            y_pred = self._get_stacked_pred()
        if y_true is None:
            y_true = self._get_stacked_gt()
        class_num = y_pred.size(1)

        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "softmax":
            activation_fn = nn.Softmax(dim=1)
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            pred_argmax = torch.argmax(y_pred, dim=1)
            activated_pred = self._one_hot(pred_argmax, y_pred, class_num)
        else:
            raise NotImplementedError("Not a supported activation!")

        gt_onehot = self._one_hot(y_true, y_pred, class_num)
        pixel_acc, dice, precision, recall = self._calculate_multi_metrics(gt_onehot, activated_pred, class_num)
        return pixel_acc, dice, precision, recall

class CosineSimLoss(nn.Module):
    def __init__(self, apply_type='channel'):
        super(CosineSimLoss, self).__init__()
        self.apply_type = apply_type #(channel, spatial)
        if apply_type == 'spatial':
            dim = 2
        else:
            dim = 1
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, output1, output2):
        if self.apply_type == 'spatial':
            similarity = self.cos(output1.reshape(output1.size(0),output1.size(1),-1), output2.reshape(output1.size(0),output1.size(1),-1))
        else:
            similarity = self.cos(output1, output2)
        loss = 1 - similarity.mean()
        return loss

class MeanSimLoss(nn.Module):
    def __init__(self, sim_type='l2'):
        super(MeanSimLoss, self).__init__()
        self.sim_type = sim_type #(channel, spatial)
        if sim_type == 'smoothl1':
            self.sim = nn.SmoothL1Loss()
        elif sim_type == 'l1':
            self.sim = nn.L1Loss()
        elif sim_type == 'cos':
            self.sim = nn.CosineSimilarity(dim=1)
        else:
            self.sim = nn.MSELoss()

    def forward(self, output1, output2):
        norm1 = output1/output1.max(-1)[0].max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        norm2 = output2/output2.max(-1)[0].max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mean_norm1 = norm1.mean(1)
        mean_norm2 = norm2.mean(1)
        if self.sim_type == 'cos':
            similarity = self.sim(mean_norm1.reshape(mean_norm1.size(0), -1),
                                  mean_norm1.reshape(mean_norm1.size(0), -1))
            loss = 1 - similarity.mean()
        else:
            loss = self.sim(mean_norm1, mean_norm2)
        return loss

class CDLoss(nn.Module):
    def __init__(self, sim_type='ce', soft_param=1, eps=1e-8):
        super(CDLoss, self).__init__()
        self.sim_type = sim_type
        self.soft_param = soft_param
        self.eps = eps

    def forward(self, output1, output2):
        prob_s = torch.exp(output1/self.soft_param)/torch.exp(output1/self.soft_param).sum((2,3), keepdim=True) + self.eps
        prob_t = torch.exp(output2/self.soft_param)/torch.exp(output2/self.soft_param).sum((2,3), keepdim=True) + self.eps
        loss = prob_t * (prob_t.log() - prob_s.log())
        return loss.sum((2,3)).mean()*(self.soft_param^2)


class BoundedRegLoss(nn.Module):
    def __init__(self, type='smooth_l1', condition_type='error', margin=0):
        super(BoundedRegLoss, self).__init__()
        self.condition_type = condition_type
        self.margin = margin
        self.loss = getattr(torch.nn.functional,f'{type}_loss')

    def forward(self, student_output, teacher_output, target):
        # Soften the student logits by applying softmax first and log() second
        if self.condition_type == 'error':
            metric_teacher = self.loss(teacher_output, target, reduce=False)
            metric_student = self.loss(student_output, target, reduce=False)
            weight = metric_teacher < (metric_student + self.margin)
        elif self.condition_type == 'error_pair':
            metric_teacher = torch.sum(self.loss(teacher_output, target, reduce=False),dim=-1, keepdim=True)
            metric_student = torch.sum(self.loss(student_output, target, reduce=False),dim=-1, keepdim=True)
            weight = (metric_teacher < (metric_student + self.margin)).expand(-1,  -1, 2)
        else:
            weight = torch.isnan(student_output).logical_not()

        loss = self.loss(student_output * weight, teacher_output * weight, reduction='sum')
        return loss / (weight.sum() + 1e-8)

class SoftCELoss(nn.Module):
    def __init__(self, T=10):
        super(SoftCELoss, self).__init__()
        self.T = T

    def forward(self, student_logits, teacher_logits, target, dim=-1):
        # Soften the student logits by applying softmax first and log() second
        soft_targets = nn.functional.softmax(teacher_logits / self.T, dim=dim)
        soft_prob = nn.functional.log_softmax(student_logits / self.T, dim=dim)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.T ** 2)
        return soft_targets_loss

class SoftBCELoss(nn.Module):
    def __init__(self, T=10., type='smooth_l1'):
        super(SoftBCELoss, self).__init__()
        self.T = T
        if type == 'focal':
            self.loss = FocalLoss()
        else:
            self.loss = getattr(torch.nn.functional,f'{type}_loss')

    def forward(self, student_logits, teacher_logits, target=None, dim=-1):
        # Soften the student logits by applying softmax first and log() second
        soft_targets = sigmoid(teacher_logits / self.T)
        soft_prob = sigmoid(student_logits / self.T)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = self.loss(soft_prob, soft_targets) * (self.T ** 2)
        return soft_targets_loss


class FocalCELoss(nn.Module):
    def __init__(self, gamma=2, reduce=True):
        super(FocalCELoss, self).__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        F_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class TemperatureFocalCELoss(nn.Module):
    def __init__(self, gamma=2, reduce=True):
        super(TemperatureFocalCELoss, self).__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets, temperature):
        """
        각 채널별 temperature가 적용된 focal CE loss
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] target labels  
            temperature: [C] 각 채널별 temperature
        """
        # temperature를 양수로 보장 (softplus 사용)
        positive_temp = F.softplus(temperature) + 1e-8  # 최소값 보장
        temp_reshaped = positive_temp.view(1, -1, 1, 1)
        
        # 각 채널별 temperature 적용
        scaled_inputs = inputs / temp_reshaped
        
        # Focal CE loss 계산
        ce_loss = nn.functional.cross_entropy(scaled_inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        F_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

# class FocalBCELoss(nn.Module):
#     def __init__(self, gamma=2, reduce=True):
#         super(FocalBCELoss, self).__init__()
#         self.gamma = gamma
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         # inputs: [B, 2, H, W] - 2 channels for binary classification
#         # targets: [B, H, W] - binary targets (0 or 1)
        
#         # Convert targets to one-hot format [B, 2, H, W]
#         targets_one_hot = torch.zeros_like(inputs)
#         targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
        
#         # Apply sigmoid to get probabilities
#         probs = torch.sigmoid(inputs)
        
#         # Calculate BCE loss for each channel
#         bce_loss = -(targets_one_hot * torch.log(probs + 1e-8) + 
#                     (1 - targets_one_hot) * torch.log(1 - probs + 1e-8))
        
#         # Calculate focal weight
#         pt = torch.where(targets_one_hot == 1, probs, 1 - probs)
#         focal_weight = (1 - pt) ** self.gamma
        
#         # Apply focal weight
#         focal_loss = focal_weight * bce_loss
        
#         if self.reduce:
#             return torch.mean(focal_loss)
#         else:
#             return focal_loss

class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, alpha_fg=0.5, alpha_bg=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha_fg = alpha_fg
        self.alpha_bg = alpha_bg
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B,2,H,W] (fg,bg logits), targets: [B,H,W] or [B,1,H,W] in {0,1}
        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()
        y_fg = targets
        y_bg = 1.0 - y_fg

        z_fg = inputs[:,0:1]
        z_bg = inputs[:,1:2]

        # CE (stable)
        ce_fg = F.binary_cross_entropy_with_logits(z_fg, y_fg, reduction="none")
        ce_bg = F.binary_cross_entropy_with_logits(z_bg, y_bg, reduction="none")

        p_fg = torch.sigmoid(z_fg); pt_fg = torch.where(y_fg>0.5, p_fg, 1-p_fg)
        p_bg = torch.sigmoid(z_bg); pt_bg = torch.where(y_bg>0.5, p_bg, 1-p_bg)

        fl_fg = (1-pt_fg).pow(self.gamma) * self.alpha_fg * ce_fg
        fl_bg = (1-pt_bg).pow(self.gamma) * self.alpha_bg * ce_bg

        loss = 0.5 * (fl_fg + fl_bg)  # 채널 평균

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class IndL1Loss1d(nn.Module):
    def __init__(self, type='l1'):
        super(IndL1Loss1d, self).__init__()
        if type == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif type == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss

    def forward(self, output, target, ind, weight):
        """ind: [b, n]"""
        output = _tranpose_and_gather_feat(output, ind)
        weight = weight.unsqueeze(2)
        loss = self.loss(output * weight, target * weight, reduction='sum')
        loss = loss / (weight.sum() * output.size(2) + 1e-4)
        return loss

# 두 선분 (p1, p2)와 (q1, q2)의 교차 여부를 판단하는 함수
def check_intersection(p1, p2, q1, q2):
    # 두 벡터가 시계 방향 또는 반시계 방향으로 정렬되어 있는지 확인
    def orientation(p, q, r):
        val = (q[..., 1] - p[..., 1]) * (r[..., 0] - q[..., 0]) - (q[..., 0] - p[..., 0]) * (r[..., 1] - q[..., 1])
        return val

    # 방향성 확인
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # 두 선분이 서로 다른 방향으로 도는 경우 교차
    intersect = (o1 * o2 < 0) & (o3 * o4 < 0)
    # print(intersect)
    return intersect


# N개의 다각형에 대해 simply connected 여부를 확인하는 함수
def check_simply_connected(polygons):
    if len(polygons.shape) == 2:
        polygons = polygons.unsqueeze(0)

    N, V, _ = polygons.shape

    # 모든 변 쌍에 대해 좌표 생성
    indices = torch.arange(V)

    # 인접하지 않은 변들만 비교 (비인접한 변이 충분히 떨어져 있는 경우만 검사)
    # 예를 들어, i와 j는 최소 2 이상의 차이를 가져야 비교
    i_indices, j_indices = torch.triu_indices(V, V, offset=2)

    # p1, p2, q1, q2는 각각 다각형의 변을 나타내는 좌표
    p1 = polygons[:, i_indices].reshape(N, -1, 2)
    p2 = polygons[:, (i_indices + 1) % V].reshape(N, -1, 2)  # % V로 인덱스가 V를 넘지 않도록 제한
    q1 = polygons[:, j_indices].reshape(N, -1, 2)
    q2 = polygons[:, (j_indices + 1) % V].reshape(N, -1, 2)

    # 모든 비인접 변 쌍에 대해 교차 여부를 확인
    intersect = check_intersection(p1, p2, q1, q2)

    # 하나라도 교차하는 경우 simple하지 않음
    is_simple = ~intersect.any(dim=1)

    return is_simple

class SelfConnectionPenalty(nn.Module):
    def __init__(self, apply_pow=True, reduce='sum'):
        super(SelfConnectionPenalty, self).__init__()
        self.apply_pow = apply_pow
        self.reduce = reduce

    # 두 벡터가 시계 방향 또는 반시계 방향으로 정렬되어 있는지 확인
    def orientation(self, p, q, r):
        val = (q[..., 1] - p[..., 1]) * (r[..., 0] - q[..., 0]) - (q[..., 0] - p[..., 0]) * (r[..., 1] - q[..., 1])
        return val

    def forward(self, polygons):
        if len(polygons.shape) == 2:
            polygons = polygons.unsqueeze(0)

        N, V, _ = polygons.shape

        # # 모든 변 쌍에 대해 좌표 생성
        # indices = torch.arange(V)

        # 인접하지 않은 변들만 비교 (비인접한 변이 충분히 떨어져 있는 경우만 검사)
        # 예를 들어, i와 j는 최소 2 이상의 차이를 가져야 비교
        i_indices, j_indices = torch.triu_indices(V, V, offset=2)

        # p1, p2, q1, q2는 각각 다각형의 변을 나타내는 좌표
        p1 = polygons[:, i_indices].reshape(N, -1, 2)
        p2 = polygons[:, (i_indices + 1) % V].reshape(N, -1, 2)  # % V로 인덱스가 V를 넘지 않도록 제한
        q1 = polygons[:, j_indices].reshape(N, -1, 2)
        q2 = polygons[:, (j_indices + 1) % V].reshape(N, -1, 2)

        # 모든 비인접 변 쌍에 대해 교차 여부를 확인
        # 방향성 확인
        o1 = self.orientation(p1, p2, q1)
        o2 = self.orientation(p1, p2, q2)
        o3 = self.orientation(q1, q2, p1)
        o4 = self.orientation(q1, q2, p2)

        # 두 선분이 서로 다른 방향으로 도는 경우 교차
        if self.apply_pow:
            penalty = torch.maximum(-(o1 * o2), torch.tensor(0, device=o1.device)).pow(2) + torch.maximum(-(o3 * o4), torch.tensor(0, device=o3.device)).pow(2)
        else:
            penalty = torch.maximum(-(o1 * o2), torch.tensor(0, device=o1.device)) + torch.maximum(-(o3 * o4), torch.tensor(0,
                                                                                                                      device=o3.device))
        return penalty.sum() if self.reduce == 'sum' else penalty.mean()

def poly_area(poly_tensor, is_polar=False):
    nb_points = poly_tensor.shape[0]

    POLAR = is_polar

    poly_tensor_polar = poly_tensor.clone()

    if POLAR:
        poly_tensor_polar[:,0], poly_tensor_polar[:,1] = torch.mul(poly_tensor[:,0], torch.cos(poly_tensor[:,1])), torch.mul(poly_tensor[:,0], torch.sin(poly_tensor[:,1]))
    #print(poly_tensor.shape)
    double = poly_tensor_polar.repeat((2,1))

    #print(double)
    polyleft = torch.mul(double[0:nb_points,0],double[1:nb_points+1,1])
    polyright = torch.mul(double[0:nb_points,1],double[1:nb_points+1,0])

    return torch.abs(0.5*(torch.sum(polyright)-torch.sum(polyleft)))

def calculate_polygon_centroid(vertices):
    n = len(vertices)

    double = vertices.repeat((2, 1))
    polyleft = torch.mul(double[0:n, 0], double[1:n + 1, 1])
    polyright = torch.mul(double[0:n, 1], double[1:n + 1, 0])
    area = torch.abs(0.5 * (torch.sum(polyright) - torch.sum(polyleft)))

    Cx = torch.sum(torch.mul(double[0:n, 0]+double[1:n+1, 0],polyleft-polyright))
    Cy = torch.sum(torch.mul(double[1:n + 1, 1]+double[0:n, 1],polyleft-polyright))
    Cx = torch.abs(Cx) / (6.0 * area)
    Cy = torch.abs(Cy) / (6.0 * area)

    return (Cx, Cy)

def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def rasterize_instances(rasterizer, instances, shape, offset=0.0):
    import itertools
    if shape[0] != shape[1]:
        raise ValueError("expected square")

    device = instances[0].gt_boxes.device
    all_polygons = clip_and_normalize_polygons(torch.from_numpy(pad_polygons(list(itertools.chain.from_iterable([
        [p[0].reshape(-1, 2) for p in inst.gt_masks.polygons] for inst in instances])))).float().to(device))

    # to me it seems the offset would need to be in _pixel_ space?
    return rasterizer(all_polygons * float(shape[1].item()) + offset, shape[1].item(), shape[0].item(), 1.0)

def clip_and_normalize_polygons(polys, inf_value=2.01):
    min_x, _ = polys[:, :, 0].min(dim=-1)
    min_y, _ = polys[:, :, 1].min(dim=-1)

    polys[torch.isinf(polys)] = -np.inf
    max_x, _ = polys[:, :, 0].max(dim=-1)
    max_y, _ = polys[:, :, 1].max(dim=-1)

    polys[torch.isinf(polys)] = inf_value

    min_xy = torch.stack((min_x, min_y), dim=-1)
    max_xy = torch.stack((max_x, max_y), dim=-1) - min_xy

    polys = (polys - min_xy.unsqueeze(1)) / max_xy.unsqueeze(1)

    return polys

def pad_polygons(polys):
    count = len(polys)
    max_vertices = max([len(p) for p in polys])
    pad_count = [max_vertices - len(p) for p in polys]

    # add between the first and second vertices.
    xs = [np.linspace(polys[i][0][0] + 0.00001, polys[i][1][0] - 0.00001, num=pad_count[i]) for i in range(count)]
    ys = [np.linspace(polys[i][0][1] + 0.00001, polys[i][1][1] - 0.00001, num=pad_count[i]) for i in range(count)]

    xys = [np.stack((xs[i], ys[i]), axis=-1) for i in range(count)]
    polys = [np.concatenate((polys[i][:1], xys[i], polys[i][1:])) for i in range(count)]

    return np.stack(polys)


class WeilPolygonClipper:

    def __init__(self, warn_if_empty=True, is_polar=False):
        self.warn_if_empty = warn_if_empty
        self.is_polar = is_polar

    def diff_cyclic(self, v1, v2, period):
        diff = v1 - v2
        if torch.is_tensor(v1) or torch.is_tensor(v2):
            diff[diff > period / 2] = diff[diff > period / 2] - period
            diff[diff < -period/2] = diff[diff < -period/2] + period
        else:
            if diff > period/2:
                diff = diff - period
            elif diff < -period/2:
                diff = diff + period
        return diff

    def is_inside(self, c1, c2, c):

        POLAR = self.is_polar

        if POLAR:
            p1 = c1[0] * torch.cos(c1[1]), c1[0] * torch.sin(c1[1])
            p2 = c2[0] * torch.cos(c2[1]), c2[0] * torch.sin(c2[1])
            q = c[0] * torch.cos(c[1]), c[0] * torch.sin(c[1])
            # R = (p2[0]*torch.cos(p2[1]) - p1[0]*torch.cos(p1[1])) * (q[0]*torch.sin(q[1]) - p1[0]*torch.sin(p1[1])) - (p2[0]*torch.sin(p2[1]) - p1[0]*torch.sin(p1[1])) * (q[0]*torch.cos(q[1]) - p1[0]*torch.cos(p1[1]))

        else:
            p1, p2, q = c1, c2, c
        R = (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])

        if R < 0:
            return 1
        elif R == 0:
            return 2
        else:
            return 0

        # return (R <= 0)

    def compute_intersection(self, c1, c2, c3, c4):

        """
        given points p1 and p2 on line L1, compute the equation of L1 in the
        format of y = m1 * x + b1. Also, given points p3 and p4 on line L2,
        compute the equation of L2 in the format of y = m2 * x + b2.

        To compute the point of intersection of the two lines, equate
        the two line equations together

        m1 * x + b1 = m2 * x + b2

        and solve for x. Once x is obtained, substitute it into one of the
        equations to obtain the value of y.

        if one of the lines is vertical, then the x-coordinate of the point of
        intersection will be the x-coordinate of the vertical line. Note that
        there is no need to check if both lines are vertical (parallel), since
        this function is only called if we know that the lines intersect.
        """
        POLAR = self.is_polar

        if POLAR:
            p1 = c1[0] * torch.cos(c1[1]), c1[0] * torch.sin(c1[1])
            p2 = c2[0] * torch.cos(c2[1]), c2[0] * torch.sin(c2[1])
            p3 = c3[0] * torch.cos(c3[1]), c3[0] * torch.sin(c3[1])
            p4 = c4[0] * torch.cos(c4[1]), c4[0] * torch.sin(c4[1])

        else:
            p1, p2, p3, p4 = c1, c2, c3, c4

        # if first line is vertical
        if p2[0] - p1[0] == 0:
            x = p1[0]

            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
            b2 = p3[1] - m2 * p3[0]

            # y-coordinate of intersection
            y = m2 * x + b2

        # if second line is vertical
        elif p4[0] - p3[0] == 0:
            x = p3[0]

            # slope and intercept of first line
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b1 = p1[1] - m1 * p1[0]

            # y-coordinate of intersection
            y = m1 * x + b1

        # if neither line is vertical
        else:
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b1 = p1[1] - m1 * p1[0]

            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
            b2 = p3[1] - m2 * p3[0]

            # x-coordinate of intersection
            x = (b2 - b1) / (m1 - m2)

            # y-coordinate of intersection
            y = m1 * x + b1

        if POLAR:
            r = torch.sqrt(x * x + y * y)
            theta = torch.atan((y + 1e-8) / (x + 1e-8))

            if x < 0:
                theta = theta + torch.pi
            elif y < 0:
                theta = theta + 2 * torch.pi

            # need to unsqueeze so torch.cat doesn't complain outside func
            intersection = torch.stack((r, theta)).unsqueeze(0)

        else:

            # need to unsqueeze so torch.cat doesn't complain outside func
            intersection = torch.stack((x, y)).unsqueeze(0)

        return intersection

    def clip(self, subject_polygon, clipping_polygon):
        # it is assumed that requires_grad = True only for clipping_polygon
        # subject_polygon and clipping_polygon are N x 2 and M x 2 torch
        # tensors respectively

        device = clipping_polygon.device

        final_polygon = torch.empty((0, 2)).to(device)

        # subject_polygon, indices = torch.sort(subject_polygon, 0)

        inters = torch.empty((0, 2)).to(device)

        # list of places for intersections (first subject then clipping then intersection)
        inbounds = []
        outbounds = []

        for i in range(len(clipping_polygon)):

            c_edge_start = clipping_polygon[i - 1]
            c_edge_end = clipping_polygon[i]

            # print('tgt', c_edge_start, c_edge_end)

            for j in range(len(subject_polygon)):

                s_edge_start = subject_polygon[j - 1]
                s_edge_end = subject_polygon[j]

                to_end = self.is_inside(c_edge_start, c_edge_end, s_edge_end)
                to_start = self.is_inside(c_edge_start, c_edge_end, s_edge_start)

                # print('pred', s_edge_start, s_edge_end)

                if to_end == 1:
                    # print('s_end inside c')
                    if to_start == 0:
                        # print('s_start outside c')
                        # Test actual intersection
                        c_in_start, c_in_end = self.is_inside(s_edge_start, s_edge_end, c_edge_end), self.is_inside(
                            s_edge_start, s_edge_end, c_edge_start)
                        if c_in_start != c_in_end:
                            # print('intersect')
                            intersection = self.compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                            inters = torch.cat((inters, intersection), dim=0)
                            inbounds.append([j, i, len(inters) - 1]) #rev14 : out>in

                elif to_start == 1:
                    # print('s_start inside c')
                    if to_end == 0:
                        # print('s_end outside c')
                        # Test actual intersection
                        c_in_start, c_in_end = self.is_inside(s_edge_start, s_edge_end, c_edge_end), self.is_inside(
                            s_edge_start, s_edge_end, c_edge_start)
                        if c_in_start != c_in_end:
                            # print('intersect')
                            intersection = self.compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                            inters = torch.cat((inters, intersection), dim=0)
                            outbounds.append([j, i, len(inters) - 1]) #rev14 : in>out

                elif to_end == 2:
                    if to_start == 0:
                        c_in_start, c_in_end = self.is_inside(s_edge_start, s_edge_end, c_edge_end), self.is_inside(
                            s_edge_start, s_edge_end, c_edge_start)
                        if c_in_start != c_in_end:
                            # print('intersect')
                            intersection = self.compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                            inters = torch.cat((inters, intersection), dim=0)
                            inbounds.append([j, i, len(inters) - 1]) #rev14 : out>in

                    elif to_start == 1:
                        c_in_start, c_in_end = self.is_inside(s_edge_start, s_edge_end, c_edge_end), self.is_inside(
                            s_edge_start, s_edge_end, c_edge_start)
                        if c_in_start != c_in_end:
                            # print('intersect')
                            intersection = self.compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                            inters = torch.cat((inters, intersection), dim=0)
                            outbounds.append([j, i, len(inters) - 1]) #rev14 : in>out

                elif to_start == 2:
                    if to_end == 0:
                        c_in_start, c_in_end = self.is_inside(s_edge_start, s_edge_end, c_edge_end), self.is_inside(
                            s_edge_start, s_edge_end, c_edge_start)
                        if c_in_start != c_in_end:
                            # print('intersect')
                            intersection = self.compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                            inters = torch.cat((inters, intersection), dim=0)
                            outbounds.append([j, i, len(inters) - 1]) #rev14 : in>out

                    elif to_end == 1:
                        c_in_start, c_in_end = self.is_inside(s_edge_start, s_edge_end, c_edge_end), self.is_inside(
                            s_edge_start, s_edge_end, c_edge_start)
                        if c_in_start != c_in_end:
                            # print('intersect')
                            intersection = self.compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                            inters = torch.cat((inters, intersection), dim=0)
                            inbounds.append([j, i, len(inters) - 1]) #rev14 : out>in

        outbounds = torch.tensor(outbounds).to(device)
        inbounds = torch.tensor(inbounds).to(device)

        # print(outbounds)
        # print(inbounds)
        # print(inters)

        used_inters = []

        while len(used_inters) < len(inbounds):
            # new_inter = 0
            inbounds = torch.roll(inbounds, -1, 0) #rev15
            new_inter = inbounds.shape[0] - 1 #rev15
            stop = inbounds[new_inter][0:2]
            j = inbounds[new_inter][0]  # j -> là où on en est dans le polygone sujet
            i = inbounds[new_inter][1]  # i -> là où on en est dans le polygone clipping
            pre_i = i #rev15
            pre_j = j #rev15

            start = True

            # Tant qu'on a pas retrouvé l'intersection de départ (aka la coord de l'intersection dans le poly sujet)
            while j != stop[0] or i != stop[1] or start:
                start = False
                flag_continue = False
                ind_select = 0
                if (torch.sum(outbounds[:, 1] >= i) == 0) and (i < (len(subject_polygon)-1)):
                    flag_except_out = True
                elif outbounds.shape[0] <= 2:
                    flag_except_out = True

                flag_except_out = False
                flag_except_in = False
                if (not flag_except_out) and (j in outbounds[:, 0]):
                    best_diff = None
                    for inds in range(torch.where(outbounds[:, 0] == j)[0].shape[0]):
                        tmp_id = torch.where(outbounds[:, 0] == j)[0][inds]
                        if not ((self.diff_cyclic(outbounds[tmp_id][1],pre_i, len(subject_polygon)) <= 0) and (self.diff_cyclic(outbounds[tmp_id][0],pre_j, len(subject_polygon)) <= 0)):
                            if best_diff is None:
                                best_diff = self.diff_cyclic(outbounds[tmp_id][1],pre_i, len(subject_polygon)) + self.diff_cyclic(outbounds[tmp_id][0],pre_j, len(subject_polygon))
                                ind_select = inds
                            elif best_diff > self.diff_cyclic(outbounds[tmp_id][1],pre_i, len(subject_polygon)) + self.diff_cyclic(outbounds[tmp_id][0],pre_j, len(subject_polygon)):
                                best_diff = self.diff_cyclic(outbounds[tmp_id][1],pre_i, len(subject_polygon)) + self.diff_cyclic(outbounds[tmp_id][0],pre_j, len(subject_polygon))
                                ind_select = inds
                            flag_continue = False
                    if best_diff is None:
                        if torch.all(self.diff_cyclic(outbounds[:,0], pre_j, len(subject_polygon)) <= 0) and torch.all(self.diff_cyclic(outbounds[:,1], pre_i, len(subject_polygon)) <= 0):
                            ind_select = (self.diff_cyclic(outbounds[outbounds[:, 0] == j,0], pre_j, len(subject_polygon)) + self.diff_cyclic(outbounds[outbounds[:, 0] == j,1], pre_i, len(subject_polygon))).max(0)[1]
                            flag_continue = False
                        else:
                            flag_continue = True
                # Tant qu'on atteint pas une intersection out (aka la coord de l'intersection dans )
                flag_incycle = True
                while ((j not in outbounds[:, 0]) or flag_continue) and flag_incycle:
                    flag_continue = False
                    final_polygon = torch.cat((final_polygon, subject_polygon[j].unsqueeze(0)), dim=0)
                    j = (j + 1) % len(subject_polygon)
                    flag_except_out = False
                    if (torch.sum(outbounds[:, 1] >= i) == 0) and (i < (len(subject_polygon) - 1)):
                        flag_except_out = True
                    elif outbounds.shape[0] <= 2:
                        flag_except_out = True

                    if flag_except_out:
                        flag_continue = False
                    elif (j in outbounds[:, 0]):
                        best_diff = None
                        for inds in range(torch.where(outbounds[:, 0] == j)[0].shape[0]):
                           if not ((self.diff_cyclic(outbounds[torch.where(outbounds[:, 0] == j)[0][inds]][1], pre_i, len(subject_polygon)) <= 0) and (self.diff_cyclic(outbounds[torch.where(outbounds[:, 0] == j)[0][inds]][0],pre_j, len(subject_polygon)) <= 0)):
                                if best_diff is None:
                                    best_diff = self.diff_cyclic(outbounds[torch.where(outbounds[:, 0] == j)[0][inds]][1], pre_i, len(subject_polygon)) + self.diff_cyclic(outbounds[torch.where(outbounds[:, 0] == j)[0][inds]][0],pre_j, len(subject_polygon))
                                    ind_select = inds
                                elif best_diff > self.diff_cyclic(outbounds[torch.where(outbounds[:, 0] == j)[0][inds]][1], pre_i, len(subject_polygon)) + self.diff_cyclic(outbounds[torch.where(outbounds[:, 0] == j)[0][inds]][0],pre_j, len(subject_polygon)):
                                    best_diff = self.diff_cyclic(outbounds[torch.where(outbounds[:, 0] == j)[0][inds]][1], pre_i, len(subject_polygon)) + self.diff_cyclic(outbounds[torch.where(outbounds[:, 0] == j)[0][inds]][0],pre_j, len(subject_polygon))
                                    ind_select = inds
                                flag_continue = False
                        if best_diff is None:
                            if torch.all(self.diff_cyclic(outbounds[:, 0], pre_j, len(subject_polygon)) <= 0) and torch.all(self.diff_cyclic(outbounds[:, 1], pre_i, len(subject_polygon)) <= 0):
                                ind_select = (self.diff_cyclic(outbounds[outbounds[:, 0] == j, 0], pre_j, len(subject_polygon)) + self.diff_cyclic(outbounds[outbounds[:, 0] == j, 1], pre_i, len(subject_polygon))).max(0)[1]
                                flag_continue = False
                            else:
                                flag_continue = True
                    flag_incycle = pre_j != j

                try:
                    new_inter = torch.where(outbounds[:, 0] == j)[0][ind_select]
                    final_polygon = torch.cat((final_polygon, inters[outbounds[new_inter][2]].unsqueeze(0)), dim=0)
                    i = outbounds[new_inter][1]
                    pre_j = outbounds[new_inter][0]
                    pre_i = i
                    outbounds = torch.vstack((outbounds[:new_inter], outbounds[new_inter + 1:]))
                except:
                    if len(inbounds) == 0:
                        break

                ## rev15
                flag_inbounds_continue = False
                if (torch.sum(inbounds[:, 0] >= j) == 0) and (j < (len(clipping_polygon) - 1)):
                    flag_except_in = True
                elif inbounds.shape[0] <= 2:
                    flag_except_in = True

                ind_select = 0
                if (not flag_except_in) and (i in inbounds[:, 1]):
                    best_diff = None
                    for inds in range(torch.where(inbounds[:, 1] == i)[0].shape[0]):
                        # ok situation -> add inter
                        if not ((0 >= self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][0], pre_j,
                                                       len(clipping_polygon)) and (
                                         0 >= self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][1],
                                                               pre_i, len(clipping_polygon))))):
                            if best_diff is None:
                                best_diff = self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][0],
                                                             pre_j, len(clipping_polygon)) + self.diff_cyclic(
                                    inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][1], pre_i,
                                    len(clipping_polygon))
                                ind_select = inds
                            elif best_diff > self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][0],
                                                              pre_j, len(clipping_polygon)) + self.diff_cyclic(
                                inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][1], pre_i,
                                len(clipping_polygon)):
                                best_diff = self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][0],
                                                             pre_j, len(clipping_polygon)) + self.diff_cyclic(
                                    inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][1], pre_i,
                                    len(clipping_polygon))
                                ind_select = inds
                            flag_inbounds_continue = False
                    if best_diff is None:
                        if torch.all(self.diff_cyclic(inbounds[:, 0], pre_j, len(clipping_polygon)) <= 0) and torch.all(
                                self.diff_cyclic(inbounds[:, 1], pre_i, len(clipping_polygon)) <= 0):
                            ind_select = (self.diff_cyclic(inbounds[inbounds[:, 1] == i, 0], pre_j,
                                                           len(clipping_polygon)) + self.diff_cyclic(
                                inbounds[inbounds[:, 1] == i, 1],
                                pre_i,
                                len(clipping_polygon))).max(
                                0)[1]
                            flag_inbounds_continue = False
                        else:
                            flag_inbounds_continue = True

                flag_incycle = True
                while ((i not in inbounds[:, 1]) or flag_inbounds_continue) and flag_incycle:
                    flag_inbounds_continue = False
                    final_polygon = torch.cat((final_polygon, clipping_polygon[i].unsqueeze(0)), dim=0)
                    i = (i + 1) % len(clipping_polygon)
                    # rev15
                    flag_except_in = False
                    if (torch.sum(inbounds[:, 0] >= j) == 0) and (j < (len(clipping_polygon) - 1)):
                        flag_except_in = True
                    elif inbounds.shape[0] <= 2:
                        flag_except_in = True
                    if (not flag_except_in) and (i in inbounds[:, 1]):
                        best_diff = None
                        for inds in range(torch.where(inbounds[:, 1] == i)[0].shape[0]):
                            if not ((0 >= self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][0],pre_j, len(clipping_polygon))) and (0 >= self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][1],pre_i, len(clipping_polygon)))):
                                if best_diff is None:
                                    best_diff = self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][0],pre_j, len(clipping_polygon)) + self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][1],pre_i, len(clipping_polygon))
                                    ind_select = inds
                                elif best_diff > self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][0],pre_j, len(clipping_polygon)) + self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][1],pre_i, len(clipping_polygon)):
                                    best_diff = self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][0],pre_j, len(clipping_polygon)) + self.diff_cyclic(inbounds[torch.where(inbounds[:, 1] == i)[0][inds]][1],pre_i, len(clipping_polygon))
                                    ind_select = inds
                                flag_inbounds_continue = False
                        if best_diff is None:
                            if torch.all(
                                    self.diff_cyclic(inbounds[:, 0], pre_j, len(clipping_polygon)) <= 0) and torch.all(
                                    self.diff_cyclic(inbounds[:, 1], pre_i, len(clipping_polygon)) <= 0):
                                ind_select = (self.diff_cyclic(inbounds[inbounds[:, 1] == i, 0], pre_j,
                                                               len(clipping_polygon)) + self.diff_cyclic(inbounds[inbounds[:, 1] == i, 1],
                                                                                                         pre_i,
                                                                                                         len(clipping_polygon))).max(
                                    0)[1]
                                flag_inbounds_continue = False
                            else:
                                flag_inbounds_continue = True
                    flag_incycle = pre_i != i

                try:
                    new_inter = torch.where(inbounds[:, 1] == i)[0][ind_select]
                    j = inbounds[new_inter][0]
                    pre_j = j
                    pre_i = inbounds[new_inter][1]
                    final_polygon = torch.cat((final_polygon, inters[inbounds[new_inter][2]].unsqueeze(0)), dim=0)
                    used_inters.append(new_inter)
                    inbounds = torch.vstack((inbounds[:new_inter], inbounds[new_inter + 1:]))
                except:
                    if len(outbounds) == 0:
                        break

        return final_polygon

    def __call__(self, A, B):
        clipped_polygon = self.clip(A, B)
        if len(clipped_polygon) == 0 and self.warn_if_empty:
            warnings.warn("No intersections found. Are you sure your polygon coordinates are in clockwise order?")

        return clipped_polygon


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class DMLoss(nn.Module):
    def __init__(self, type='l1'):
        type_list = {'l1': torch.nn.functional.l1_loss, 'smooth_l1': torch.nn.functional.smooth_l1_loss}
        self.crit = type_list[type]
        super(DMLoss, self).__init__()

    def interpolation(self, poly, time=10):
        ori_points_num = poly.size(1)
        poly_roll =torch.roll(poly, shifts=1, dims=1)
        poly_ = poly.unsqueeze(3).repeat(1, 1, 1, time)
        poly_roll = poly_roll.unsqueeze(3).repeat(1, 1, 1, time)
        step = torch.arange(0, time, dtype=torch.float32).cuda() / time
        poly_interpolation = poly_ * step + poly_roll * (1. - step)
        poly_interpolation = poly_interpolation.permute(0, 1, 3, 2).reshape(poly_interpolation.size(0), ori_points_num * time, 2)
        return poly_interpolation

    def compute_distance(self, pred_poly, gt_poly):
        pred_poly_expand = pred_poly.unsqueeze(1)
        gt_poly_expand = gt_poly.unsqueeze(2)
        gt_poly_expand = gt_poly_expand.expand(gt_poly_expand.size(0), gt_poly_expand.size(1),
                                               pred_poly_expand.size(2), gt_poly_expand.size(3))
        pred_poly_expand = pred_poly_expand.expand(pred_poly_expand.size(0), gt_poly_expand.size(1),
                                                   pred_poly_expand.size(2), pred_poly_expand.size(3))
        distance = torch.sum((pred_poly_expand - gt_poly_expand) ** 2, dim=3)
        return distance
    
    def lossPred2NearestGt(self, ini_pred_poly, pred_poly, gt_poly, all_loss={}):
        gt_poly_interpolation = self.interpolation(gt_poly)
        distance_pred_gtInterpolation = self.compute_distance(ini_pred_poly, gt_poly_interpolation)
        index_gt = torch.min(distance_pred_gtInterpolation, dim=1)[1]
        index_0 = torch.arange(index_gt.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_gt.size(0), index_gt.size(1))
        loss_predto_nearestgt = self.crit(pred_poly,gt_poly_interpolation[index_0, index_gt, :], reduction='none')
        all_loss.update({'dml_loss_predto_nearestgt': loss_predto_nearestgt})
        return loss_predto_nearestgt.mean()

    def lossGt2NearestPred(self, ini_pred_poly, pred_poly, gt_poly):
        distance_pred_gt = self.compute_distance(ini_pred_poly, gt_poly)
        index_pred = torch.min(distance_pred_gt, dim=2)[1]
        index_0 = torch.arange(index_pred.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_pred.size(0), index_pred.size(1))
        loss_gtto_nearestpred = self.crit(pred_poly[index_0, index_pred, :], gt_poly,reduction='none')
        return loss_gtto_nearestpred

    def setloss(self, ini_pred_poly, pred_poly, gt_poly, keyPointsMask, all_loss={}):
        keyPointsMask = keyPointsMask.unsqueeze(2).expand(keyPointsMask.size(0), keyPointsMask.size(1), 2)
        lossPred2NearestGt = self.lossPred2NearestGt(ini_pred_poly, pred_poly, gt_poly, all_loss=all_loss)
        lossGt2NearestPred = self.lossGt2NearestPred(ini_pred_poly, pred_poly, gt_poly)

        all_loss.update({'dml_lossGt2NearestPred' : lossGt2NearestPred, 'dml_keyPointsMask': keyPointsMask})
        loss_set2set = torch.sum(lossGt2NearestPred * keyPointsMask) / (torch.sum(keyPointsMask) + 1) + lossPred2NearestGt
        return loss_set2set / 2.

    def forward(self, ini_pred_poly, pred_polys_, gt_polys, keyPointsMask, all_loss={}):
        return self.setloss(ini_pred_poly, pred_polys_, gt_polys, keyPointsMask, all_loss=all_loss)


class TVLoss(nn.Module):
    def __init__(self, type='l1'):
        type_list = {'l1': torch.nn.functional.l1_loss, 'smooth_l1': torch.nn.functional.smooth_l1_loss,
                     'l2': torch.nn.functional.mse_loss}
        self.crit = type_list[type]
        super(TVLoss, self).__init__()

    def forward(self, pred_polys_):
        '''
        pred_polys_ : N_py x N_vert x 2
        '''
        poly_roll = torch.roll(pred_polys_, shifts=1, dims=1)
        return self.crit(poly_roll, pred_polys_)

class CurvLoss(nn.Module):
    def __init__(self, type='l1'):
        type_list = {'l1': torch.nn.functional.l1_loss, 'smooth_l1': torch.nn.functional.smooth_l1_loss,
                     'l2': torch.nn.functional.mse_loss}
        self.crit = type_list[type]
        super(CurvLoss, self).__init__()

    def forward(self, pred_polys_):
        '''
        pred_polys_ : N_py x N_vert x 2
        '''
        poly_pre = torch.roll(pred_polys_, shifts=1, dims=1)
        poly_post = torch.roll(pred_polys_, shifts=-1, dims=1)

        return self.crit(poly_pre+poly_post, 2*pred_polys_)


class WeightedPYLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self, type='weighted_smooth_l1', th_weight=3., under_weight=0.):
        super(WeightedPYLoss, self).__init__()
        type_list = {'weighted_l1': torch.nn.functional.l1_loss, 'weighted_smooth_l1': torch.nn.functional.smooth_l1_loss,
                     'weighted_l2': torch.nn.functional.mse_loss}
        self.crit = type_list[type]
        self.th_weight = th_weight
        self.under_weight = under_weight

    def forward(self, out, target):
        crit = self.crit(out, target, reduction='none')
        weight_mask = crit >= self.th_weight
        under_mask = crit < self.th_weight
        union = self.under_weight * under_mask.clone().detach() * crit + weight_mask.clone().detach() * crit
        return union.mean()


class MDLoss(nn.Module):
    def __init__(self, type='l1', match_with_ini=True):
        '''
        Minimum Distance Loss
        :param type:
        '''
        type_list = {'l1': torch.nn.functional.l1_loss,
                     'smooth_l1': torch.nn.functional.smooth_l1_loss}
        self.crit = type_list[type]
        self.match_with_ini = match_with_ini
        super(MDLoss, self).__init__()

    def compute_distance(self, pred_poly, gt_poly):
        pred_poly_expand = pred_poly.unsqueeze(1) #(N,1,L,2)
        gt_poly_expand = gt_poly.unsqueeze(2) #(N,L,1,2)
        gt_poly_expand = gt_poly_expand.expand(gt_poly_expand.size(0),
                                               gt_poly_expand.size(1),
                                               pred_poly_expand.size(2),
                                               gt_poly_expand.size(3)) #(N,L,1,2) -> (N,L,L,2)
        pred_poly_expand = pred_poly_expand.expand(pred_poly_expand.size(0),
                                                   gt_poly_expand.size(1),
                                                   pred_poly_expand.size(2),
                                                   pred_poly_expand.size(3)) #(N,1,L,2) -> (N,L,L,2)
        distance = torch.sum((pred_poly_expand - gt_poly_expand) ** 2, dim=3) #(N,L,L) : gtxpred
        return distance

    def lossPred2NearestGt(self, ini_pred_poly, pred_poly, gt_poly,
                           all_loss={}):
        if self.match_with_ini:
            distance_pred_gt = self.compute_distance(ini_pred_poly,
                                                                  gt_poly)#(N,L,L)
        else:
            distance_pred_gt = self.compute_distance(pred_poly,
                                                     gt_poly)  # (N,L,L)
        index_gt = torch.min(distance_pred_gt, dim=1)[1]#(N,[L],L)
        # index_0 = torch.arange(index_gt.size(0)) #(0...N-1)
        # index_0 = index_0.unsqueeze(1).expand(index_gt.size(0),
        #                                       index_gt.size(1)) #(N,1) -> (N,L)
        gt_poly_matched = torch.gather(gt_poly, 1, index_gt.unsqueeze(-1).expand(-1, -1, 2).to(torch.int64))
        loss_predto_nearestgt = self.crit(pred_poly,
                                          gt_poly_matched, reduction='none')
        all_loss.update({'pred_poly': pred_poly, 'gt_poly_matched': gt_poly_matched, 'index_gt': index_gt})
        all_loss.update({'mdl_loss_predto_nearestgt': loss_predto_nearestgt})
        return loss_predto_nearestgt.mean()


    # def setloss(self, ini_pred_poly, pred_poly, gt_poly, keyPointsMask,
    #             all_loss={}):
    #     # keyPointsMask = keyPointsMask.unsqueeze(2).expand(keyPointsMask.size(0),
    #     #                                                   keyPointsMask.size(1),
    #     #                                                   2)
    #     lossPred2NearestGt = self.lossPred2NearestGt(ini_pred_poly, pred_poly,
    #                                                  gt_poly, all_loss=all_loss)
    #     # lossGt2NearestPred = self.lossGt2NearestPred(ini_pred_poly, pred_poly,
    #     #                                              gt_poly)
    #
    #     # all_loss.update({'mdl_keyPointsMask': keyPointsMask})
    #     # loss_set2set = lossPred2NearestGt
    #     return lossPred2NearestGt / 2.

    def forward(self, ini_pred_poly, pred_polys_, gt_polys, all_loss={}):
        lossPred2NearestGt = self.lossPred2NearestGt(ini_pred_poly, pred_polys_,
                                                     gt_polys, all_loss=all_loss)
        return lossPred2NearestGt

@torch.no_grad()
def label_to_one_hot_label(
        labels: torch.Tensor,
        num_classes: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
        ignore_index=255,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([
                [[0, 1],
                [2, 0]]
            ])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],

                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],

                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    shape = labels.shape
    # one hot : (B, C=ignore_index+1, H, W)
    one_hot = torch.zeros((shape[0], ignore_index + 1) + shape[1:], device=device, dtype=dtype, requires_grad=False)

    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # one_hot : (B, C=ignore_index+1, H, W)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

    # ret : (B, C=num_classes, H, W)
    ret = torch.split(one_hot, [num_classes, ignore_index + 1 - num_classes], dim=1)[0]

    return ret

# https://github.com/zhezh/focalloss/blob/master/focalloss.py
def focal_loss(input, target, alpha, gamma, reduction, eps, ignore_index):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0)  # B

    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]

    # input : (B, C, H, W)
    # target : (B, H, W)
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)

        # compute softmax over the classes axis
    # input_soft : (B, C, H, W)
    input_soft = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    # target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device,
    #                                         dtype=input.dtype, ignore_index=ignore_index)

    target_one_hot = nn.functional.one_hot(target.long(), num_classes=input.shape[1])
    dims_target = [0, len(target_one_hot.shape)-1]
    for dimi in range(1, len(target_one_hot.shape)-1):
        dims_target.append(dimi)
    target_one_hot = target_one_hot.permute(tuple(dims_target))

    # compute the actual focal loss
    weight = torch.pow(1.0 - input_soft, gamma)

    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(input_soft)

    # loss_tmp : (B, H, W)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class MultiFocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math:

        FL(p_t) = -alpha_t(1 - p_t)^{gamma}, log(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha, gamma=2.0, reduction='mean', eps=1e-8, ignore_index=30):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index)


class mIoULoss(torch.nn.Module):
    def __init__(self, n_classes=4, device='cuda'):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.device = device

    def forward(self, inputs, target, apply_softmax=True):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        inputs = inputs.to(self.device)
        target = target.to(self.device)

        SMOOTH = 1e-6
        N = inputs.size()[0]

        if apply_softmax:
            inputs = F.softmax(inputs, dim=1)
        # target_oneHot = label_to_one_hot_label(target, self.classes, device=self.device)
        target_oneHot = nn.functional.one_hot(target.long(), num_classes=self.classes)
        dims_target = [0, len(target_oneHot.shape) - 1]
        for dimi in range(1, len(target_oneHot.shape) - 1):
            dims_target.append(dimi)
        target_oneHot = target_oneHot.permute(tuple(dims_target))
        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2) + SMOOTH

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2) + SMOOTH

        loss = inter / union

        ## Return average loss over classes and batch
        return (1-loss).mean()


class PolyLoss(nn.Module):
    def __init__(self, cfg=None):
        super(PolyLoss, self).__init__()
        self.cfg = cfg
        self.clip = WeilPolygonClipper()
        if cfg.train.iou_params['use_keypoints']:
            self.Douglas = DouglasTorch(num_vertices=cfg.train.iou_params['num_keypoints'] if 'num_keypoints' in cfg.train.iou_params else None,
                                        D=cfg.data.douglas['D'] if 'D' in cfg.data.douglas else 3, extract_type=cfg.data.douglas['extract_type'] if 'extract_type' in cfg.data.douglas else None)
            self.DouglasGT = self.Douglas
            if 'split_gt_objects' in cfg.data.douglas:
                if cfg.data.douglas['split_gt_objects']:
                    self.DouglasGT = DouglasTorch(
                        num_vertices=cfg.train.iou_params['num_keypoints_gt'] if 'num_keypoints_gt' in cfg.train.iou_params else None,
                        D=cfg.data.douglas['D_gt'] if 'D_gt' in cfg.data.douglas else 0.01,
                        extract_type=cfg.data.douglas['extract_type_gt'] if 'extract_type_gt' in cfg.data.douglas else None)
        else:
            self.Douglas = None
            self.DouglasGT = None

    def forward(self, pred, target, keypointsmask=None, epoch=None, save_mode=False):
        """
        Parameters:
            output: output of polygon head
              [batch_size, 2*nb_vertices, height, width]
            mask: selected objects
              [batch_size, nb_max_objects]
            ind:
              [batch_size, nb_max_objects]
            target: ground-truth for the polygons
              [batch_size, nb_max_objects, 2*nb_vertices]
            hm: output of heatmap head
              [batch_size, nb_categories, height, width]
        Returns:
            loss: scalar
        """

        # pred = _transpose_and_gather_feat(output, ind)

        save_dict = {}
        if save_mode:
            save_dict['sorted_pred'] = []
            save_dict['sorted_target'] = []
            save_dict['clipped_polygon'] = []
            save_dict['area_target'] = []
            save_dict['area_pred'] = []
            save_dict['area_intersection'] = []
            save_dict['intersection'] = []
            save_dict['union'] = []
            save_dict['valid_region'] = []
        loss = torch.tensor(0., device=pred.device)
        n_sample = 0.

        #rev22: 24-09-08
        # if self.cfg.train.iou_params['filter_type'] == 'is_simple':
        #     is_py_simple = check_simply_connected(pred)
        # else:
        #     is_py_simple = [True for i in range(pred.shape[0])]

        for i in range(pred.shape[0]):  # nbr objects
            #How to sort: get indices when sorting angle columns and apply it to the whole tensor
            # sorted_pred = pred[batch][i].view(-1,2)[torch.sort(pred[batch][i].view(-1,2)[:,1],0)[1]]
            if self.Douglas is not None:
                sorted_pred = pred[i,...][self.Douglas.sample(pred[i,...]).bool()] #rev14-remove flip[0]
                if keypointsmask is None:
                    keymask = self.DouglasGT.sample(target[i, ...])
                else:
                    keymask = keypointsmask[i]
                sorted_target = target[i, ...][keymask.bool()] #rev14-remove flip[0]
            else:
                sorted_pred = pred[i,...] #rev14-remove flip[0]
                sorted_target = target[i, ...] #rev14-remove flip[0]

            # rev23: 24-09-10
            if ('rollback_gamma' not in self.cfg.train.optimizer) and (self.cfg.train.iou_params['filter_type'] == 'is_simple'):
                is_py_simple = check_simply_connected(sorted_pred)
            else:
                is_py_simple = True

            if is_py_simple:
                # sorted_pred = torch.cat((sorted_pred[:,0].unsqueeze(1), sorted_pred[:,1].unsqueeze(1)), 1) #n_vertx2
                clipped_polygon = self.clip(sorted_pred, sorted_target) #.flip(0)) #rev11-add cat #rev13-remove cat

                area_target = poly_area(sorted_target)
                area_pred = poly_area(sorted_pred)
                area_intersection = poly_area(clipped_polygon)
                intersection = (area_intersection.item() == 0.0)*torch.min(area_pred,area_target)+area_intersection
                union = area_target + area_pred - intersection

                if save_mode:
                    np_sorted_pred = sorted_pred.clone().detach().cpu().numpy()
                    save_dict['sorted_pred'].append(np_sorted_pred)
                    np_sorted_target = sorted_target.clone().detach().cpu().numpy()
                    save_dict['sorted_target'].append(np_sorted_target)
                    np_clip_polygon = clipped_polygon.clone().detach().cpu().numpy()
                    save_dict['clipped_polygon'].append(np_clip_polygon)
                    save_dict['area_target'].append(area_target.clone().detach().cpu().numpy())
                    save_dict['area_pred'].append(area_pred.clone().detach().cpu().numpy())
                    save_dict['area_intersection'].append(area_intersection.clone().detach().cpu().numpy())
                    save_dict['intersection'].append(intersection.clone().detach().cpu().numpy())
                    save_dict['union'].append(union.clone().detach().cpu().numpy())

                if (epoch is not None) and (i in (0, )):
                    from scipy.io import savemat
                    import os
                    os.makedirs(f"{self.cfg.commen.result_dir}/OnTraining/Polyloss", exist_ok=True)
                    savemat(f"{self.cfg.commen.result_dir}/OnTraining/Polyloss/e{epoch}_{i}.mat",
                            {'pred_origin': pred[i, ...].clone().detach().cpu().numpy(),
                             'target_origin': target[i, ...].clone().detach().cpu().numpy(),
                             'pred': sorted_pred.clone().detach().cpu().numpy(),
                             'target': sorted_target.clone().detach().cpu().numpy(),
                             'clipped_polygon': clipped_polygon.clone().detach().cpu().numpy(),
                             'area_target' : area_target.clone().detach().cpu().numpy(),
                             'area_pred': area_pred.clone().detach().cpu().numpy(),
                             'area_intersection': area_intersection.clone().detach().cpu().numpy(),
                             'intersection': intersection.clone().detach().cpu().numpy(),
                             'union': union.clone().detach().cpu().numpy()})

                # rev 24-07-17 20:17
                if self.cfg.train.iou_params['filter_type'] == 'iou_val':
                    if (intersection/(union+ 1e-6) <= 1.) and (intersection/(union+ 1e-6) >= 0.):
                        loss += intersection/(union+ 1e-6)
                        n_sample += 1
                        if save_mode:
                            save_dict['valid_region'].append(1)
                    elif save_mode:
                        save_dict['valid_region'].append(0)
                else:
                    loss += intersection / (union + 1e-6)
                    n_sample += 1
                    if save_mode:
                        save_dict['valid_region'].append(1)
            else:
                if save_mode:
                    save_dict['valid_region'].append(0)

        loss = (1 - loss / (n_sample + 1e-6))

        #print(loss)
        #print("------------")

        return loss, save_dict, is_py_simple


class DouglasTorch:
    def __init__(self, D=3, num_vertices=None, extract_type=None):
        self.D = D
        self.num_vertices = num_vertices
        if extract_type is None:
            self.extract_type = 'must'
        else:
            self.extract_type = extract_type

    def get_uniformly_spaced_additional_numbers(self, initial, total, count):
        # 모든 숫자 집합을 정렬
        all_sorted_indices = [x for x in range(total) if x not in initial]

        # 전체 숫자 범위에서 균일한 간격을 유지하도록 구간을 나누기
        intervals = np.linspace(0, len(all_sorted_indices), count + 1)

        selected = initial.clone()
        additional = []

        not_include = 0
        for i in range(count):
            segment_start = int(intervals[i])
            segment_end = int(intervals[i + 1])
            possible_values = [x for x in all_sorted_indices[segment_start:segment_end] if x not in selected]

            if possible_values:
                chosen_value = np.random.choice(possible_values)
                additional.append(chosen_value)
                selected = torch.cat([selected,torch.tensor(chosen_value,device=selected.device).unsqueeze(0)])
            else:
                not_include += 1

        if not_include > 0:
            sorted_selected = torch.tensor(sorted(selected),device=initial.device)
            diff = torch.roll(sorted_selected,-1) - sorted_selected
            not_include_selected = torch.randint(sorted_selected[diff.max(0)[1]]+1, sorted_selected[diff.max(0)[1]+1], size=not_include)
            additional.extend(list(not_include_selected))
        return additional

    def find_best_starting_point(self, poly):
        centroid_coords = calculate_polygon_centroid(poly)
        distances = torch.sum(torch.abs(poly-torch.tensor(centroid_coords, device=poly.device)),-1)
        closest_ind = distances.min(0)[1].item()
        return closest_ind, centroid_coords

    @torch.no_grad()
    def sample(self, poly):
        mask = torch.zeros((poly.shape[0],), dtype=torch.int32).to(poly.device)
        distance = torch.zeros((poly.shape[0],), dtype=torch.int32).to(poly.device)
        # mask[0] = 1 #rev7
        # endPoint = poly[0, :] + poly[-1:, :]
        # endPoint /= 2
        # poly_append = torch.cat([poly, endPoint], dim=0)
        # start_ind = 0 #rev7
        # start_ind = random.randint(0, poly.shape[0]-1) #rev8
        start_ind, centroid_coords = self.find_best_starting_point(poly) #rev16
        poly_append = torch.roll(poly, -start_ind, 0)  # roll(-start_ind), rev8
        d = torch.sum(torch.abs(poly_append[0,:] - poly_append), dim=-1)
        max_idx = torch.argmax(d)
        dmax = d[max_idx]
        distance[max_idx] = dmax

        self.compress(0, max_idx, poly_append, mask, distance)
        self.compress(max_idx, poly_append.shape[0]-1, poly_append, mask, distance)
        if self.num_vertices is not None:
            # [d_vals, d_inds] = torch.topk(distance, self.num_vertices-1) #rev6
            [d_vals, d_inds] = torch.topk(distance, self.num_vertices)  # rev7
            mask[:] = 0
            # mask[0] = 1#rev7
            mask[d_inds[d_vals > 0]] = 1
            if self.extract_type == 'must':
                add_inds = self.get_uniformly_spaced_additional_numbers(d_inds[d_vals > 0], len(mask), self.num_vertices-mask.sum()) #rev15
                mask[add_inds] = 1
            # if mask.sum() != self.num_vertices:
            #     print(mask.sum(), len(add_inds), 'pre : ', len(d_inds[d_vals > 0]))
            #     print(mask[d_inds[d_vals > 0]].sum(), mask[add_inds].sum())
            #     print(f"intersection  :{set(list(d_inds[d_vals > 0])).intersection(set(list(add_inds)))}")
            #     print(f"add_inds : {add_inds}")
            #     print(f"d_inds[d_vals > 0] : {d_inds[d_vals > 0]}")

            # print(f"d_inds : {d_inds} / mask : {mask}")
        mask = torch.roll(mask, start_ind, 0) #rev8
        return mask

    @torch.no_grad()
    def compress(self, idx1, idx2, poly, mask, distance):
        p1 = poly[idx1, :]
        p2 = poly[idx2, :]
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])

        m = idx1
        n = idx2
        if (n <= m + 1):
            return
        d = torch.abs(A * poly[m + 1: n, 0] + B * poly[m + 1: n, 1] + C) / torch.sqrt(torch.pow(A, 2) + torch.pow(B, 2) + 1e-4)
        max_idx = torch.argmax(d)
        dmax = d[max_idx]
        max_idx = max_idx + m + 1

        if dmax > self.D:
            mask[max_idx] = 1
            distance[max_idx] = dmax
            self.compress(idx1, max_idx, poly, mask, distance)
            self.compress(max_idx, idx2, poly, mask, distance)
        # elif self.num_vertices is not None:
        #     if mask.sum() < self.num_vertices:
        #         mask[max_idx] = 1
        #         self.compress(idx1, max_idx, poly, mask)
        #         self.compress(max_idx, idx2, poly, mask)


# standard deviation loss
class EdgeStandardDeviationLoss(nn.Module):
    def __init__(self, cfg=None):
        super(EdgeStandardDeviationLoss, self).__init__()
        self.cfg = cfg

    def forward(self, polys):
        '''
        :param polys: (N_contour, N_vertex, 2)
        :return:
        '''
        poly_roll = torch.roll(polys, shifts=1, dims=1)
        edges = (poly_roll-polys).pow(2).sum(-1).sqrt() #(N_contour, N_vertex)
        mean_edges = edges.mean(-1) #(N_contour,)
        loss = torch.sqrt((edges-mean_edges.unsqueeze(-1)).pow(2).sum(-1).sqrt().sum()/polys.size(0))
        return loss


# ==============================================================================
# Advanced Pixel Loss Functions for Stage 1 Training
# ==============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: (B, H, W) or (B, 1, H, W) - predicted probability after sigmoid
        target: (B, H, W) - binary ground truth
        """
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
            
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss with adjustable FP/FN weights."""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta    # weight for false negatives
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: (B, H, W) or (B, 1, H, W) - predicted probability after sigmoid
        target: (B, H, W) - binary ground truth
        """
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
            
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky


class BoundaryLoss(nn.Module):
    """Boundary-aware loss focusing on edge regions."""
    def __init__(self, boundary_weight=2.0, smooth=1e-6):
        super(BoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.smooth = smooth
        
    def get_boundary_mask(self, target):
        """Extract boundary pixels from target mask."""
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=target.dtype, device=target.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=target.dtype, device=target.device).view(1, 1, 3, 3)
        
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B, 1, H, W)
            
        edge_x = F.conv2d(target.float(), sobel_x, padding=1)
        edge_y = F.conv2d(target.float(), sobel_y, padding=1)
        edge = torch.sqrt(edge_x.pow(2) + edge_y.pow(2))
        boundary = (edge > 0.1).float().squeeze(1)  # (B, H, W)
        return boundary

    def forward(self, pred, target):
        """
        pred: (B, H, W) or (B, 1, H, W) - predicted probability after sigmoid
        target: (B, H, W) - binary ground truth
        """
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
            
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy(pred, target.float(), reduction='none')
        
        # Boundary mask
        boundary_mask = self.get_boundary_mask(target)
        
        # Weighted loss: higher weight for boundary pixels
        weight_mask = 1.0 + self.boundary_weight * boundary_mask
        weighted_loss = bce_loss * weight_mask
        
        return weighted_loss.mean()


class ComboLoss(nn.Module):
    """Combination of multiple loss functions for better pixel segmentation."""
    def __init__(self, focal_weight=0.5, dice_weight=0.3, tversky_weight=0.2, 
                 focal_alpha=0.25, focal_gamma=2.0, tversky_alpha=0.7):
        super(ComboLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, logits=False)
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha)

    def forward(self, pred, target):
        """
        pred: (B, H, W) or (B, 1, H, W) - predicted probability after sigmoid
        target: (B, H, W) - binary ground truth
        """
        focal = self.focal_loss(pred, target.float())
        dice = self.dice_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        
        combo = (self.focal_weight * focal + 
                self.dice_weight * dice + 
                self.tversky_weight * tversky)
        return combo


class AdaptivePixelLoss(nn.Module):
    """Adaptive pixel loss that adjusts based on class imbalance."""
    def __init__(self, base_loss='focal', adaptive_weight=True, min_weight=0.1, max_weight=10.0):
        super(AdaptivePixelLoss, self).__init__()
        self.adaptive_weight = adaptive_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        if base_loss == 'focal':
            self.base_loss = FocalLoss(logits=False)
        elif base_loss == 'dice':
            self.base_loss = DiceLoss()
        elif base_loss == 'tversky':
            self.base_loss = TverskyLoss()
        elif base_loss == 'combo':
            self.base_loss = ComboLoss()
        else:
            self.base_loss = nn.BCELoss()

    def forward(self, pred, target):
        """
        pred: (B, H, W) or (B, 1, H, W) - predicted probability after sigmoid
        target: (B, H, W) - binary ground truth
        """
        loss = self.base_loss(pred, target)
        
        if self.adaptive_weight:
            # Calculate class ratio for adaptive weighting
            if target.dim() == 4:
                target_flat = target.squeeze(1)
            else:
                target_flat = target
                
            pos_ratio = target_flat.float().mean()
            neg_ratio = 1.0 - pos_ratio
            
            # Avoid division by zero and extreme weights
            pos_ratio = torch.clamp(pos_ratio, min=0.01, max=0.99)
            neg_ratio = torch.clamp(neg_ratio, min=0.01, max=0.99)
            
            # Adaptive weight based on class imbalance
            if pos_ratio < 0.1:  # Very few positive pixels
                weight = torch.clamp(1.0 / pos_ratio, min=self.min_weight, max=self.max_weight)
                loss = loss * weight
            elif neg_ratio < 0.1:  # Very few negative pixels
                weight = torch.clamp(1.0 / neg_ratio, min=self.min_weight, max=self.max_weight)
                loss = loss * weight
                
        return loss
