import torch.nn as nn
import torch
import torch.nn.functional as F

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

def topk(scores, K=100):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def decode_ct_hm_snake(ct_hm, wh, reg=None, K=100, stride=10.):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)

    scores, inds, clses, ys, xs = topk(ct_hm, K=K)
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)

    detection = torch.cat([bboxes, scores, clses], dim=2)
    return ct, detection

def decode_ct_hm(ct_hm, wh, reg=None, K=100, stride=10.):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)
    scores, inds, clses, ys, xs = topk(ct_hm, K=K)
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, -1, 2)

    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    ct = torch.cat([xs, ys], dim=2)
    poly = ct.unsqueeze(2).expand(batch, K, wh.size(2), 2) + wh * stride
    detection = torch.cat([ct, scores, clses], dim=2)
    return poly, detection, wh

def clip_to_image(poly, h, w):
    poly[..., :2] = torch.clamp(poly[..., :2], min=0)
    poly[..., 0] = torch.clamp(poly[..., 0], max=w-1)
    poly[..., 1] = torch.clamp(poly[..., 1], max=h-1)
    return poly

# def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
#     img_poly = img_poly.clone()
#     img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
#     img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
#     batch_size = cnn_feature.size(0)
#     if img_poly.size(0) == 0:
#         return torch.empty(0, cnn_feature.size(1), 1, device=cnn_feature.device)
#     gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
#     for i in range(batch_size):
#         poly = img_poly[ind == i].unsqueeze(0)
#         feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
#         gcn_feature[ind == i] = feature
#     return gcn_feature

# def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
#     """
#     Args:
#         cnn_feature: [B, C, H, W]
#         img_poly: [N_poly, N_vert, 2] in pixel coords
#         ind: [N_poly] — image indices (0 ~ B-1)
#         h, w: height and width of feature map
#
#     Returns:
#         gcn_feature: [N_poly, C, N_vert]
#     """
#     # clone & normalize polygon coordinates to [-1, 1]
#     img_poly = img_poly.clone()
#     img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
#     img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
#
#     batch_size = cnn_feature.size(0)
#     num_polys = img_poly.size(0)
#     num_verts = img_poly.size(1)
#     C = cnn_feature.size(1)
#
#     # Handle empty input (no polygons)
#     if num_polys == 0:
#         print("[Warning] get_gcn_feature received empty polygon tensor.")
#         return torch.empty(0, C, 1, device=cnn_feature.device)
#
#     gcn_feature = torch.zeros((num_polys, C, num_verts), device=cnn_feature.device)
#
#     for i in range(batch_size):
#         mask = (ind == i)
#         if mask.sum() == 0:
#             continue  # skip if no polygon for this image
#         poly = img_poly[mask].unsqueeze(0)  # [1, N_poly_i, N_vert, 2]
#         sampled = F.grid_sample(cnn_feature[i:i+1], poly, align_corners=True)  # [1, C, N_vert, N_poly_i]
#         gcn_feature[mask] = sampled.squeeze(0).permute(2, 0, 1)  # [N_poly_i, C, N_vert]
#
#     return gcn_feature
def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    """
    Args:
        cnn_feature: [B, C, H, W]
        img_poly: [N_poly, N_vert, 2] (pixel coords)
        ind: [N_poly] (image indices for each poly)
        h, w: height and width of feature map

    Returns:
        gcn_feature: [N_poly, C, N_vert]
    """
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    B, C = cnn_feature.shape[:2]
    N_poly = img_poly.size(0)
    N_vert = img_poly.size(1)

    if N_poly == 0:
        return torch.empty(0, C, 1, device=cnn_feature.device)

    gcn_feature = torch.zeros((N_poly, C, N_vert), device=cnn_feature.device)

    for i in range(B):
        mask = (ind == i)
        if mask.sum() == 0:
            continue
        poly = img_poly[mask].unsqueeze(0)  # [1, N_poly_i, N_vert, 2]
        poly = poly.permute(0, 2, 1, 3)      # -> [1, N_vert, N_poly_i, 2]
        sampled = F.grid_sample(cnn_feature[i:i+1], poly, align_corners=True)

        feat = sampled.squeeze(0).permute(2, 0, 1)  # -> [N_poly_i, C, N_vert]
        gcn_feature[mask] = feat

    return gcn_feature

def uniform_upsample(poly, p_num):
    from network.extreme_utils_replacement import uniform_upsample_pure
    return uniform_upsample_pure(poly, p_num)

def global_to_local_ct_img_idx(global_idx, device_id, world_size, total_batch_size):
    """
    Convert global index to local index in DataParallel environment.

    Example:
        When total batch_size is 32 and 4 GPUs are used,
        each GPU processes 8 samples and
        global index 9 becomes local index 1 on device 1.

    Args:
        global_idx (Tensor): global index of shape [N] (e.g., ct_img_idx)
        device_id (int): current device ID (e.g., 0, 1, 2...)
        world_size (int): 총 디바이스 수
        total_batch_size (int): 전체 배치 크기 (예: 32, 64 등)

    Returns:
        Tensor: local index (global_idx - 시작 인덱스)
    """
    if global_idx.numel() == 0:
        return global_idx  # 빈 tensor 처리

    local_batch_size = total_batch_size // world_size
    local_start = device_id * local_batch_size
    local_idx = global_idx - local_start

    # 안전하게 클리핑 (음수 또는 범위 초과 방지)
    local_idx = torch.clamp(local_idx, min=0, max=local_batch_size - 1)

    return local_idx