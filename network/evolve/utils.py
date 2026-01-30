import torch
from network.detector_decode.utils import uniform_upsample
from network import data_utils
import numpy as np
import torch.nn.functional as F

def global_to_local_ct_img_idx(ct_img_idx: torch.Tensor,
                               device_id: int,
                               world_size: int,
                               global_batch_size: int,
                               return_mask: bool = True):
    """
    DataParallel(DP)에서 batch dim이 world_size로 균등/라운드-로빈 분할된다고 가정하고,
    글로벌 배치 인덱스(0..global_B-1)를 이 디바이스의 로컬 인덱스(0..local_B-1)로 변환.

    Args:
        ct_img_idx: (M,) long tensor, 글로벌 배치 인덱스
        device_id: 이 replica의 디바이스 ID (예: 0..world_size-1)
        world_size: GPU 개수
        global_batch_size: 한 스텝의 글로벌 배치 사이즈 (모든 GPU 합)
        return_mask: True면 (local_idx, keep_mask) 반환

    Returns:
        local_idx: (M_keep,) long  — 이 샤드에 속하는 것만, start를 빼서 0..local_B-1로 변환
        keep_mask: (M,) bool      — 이 샤드에 속하는 위치(True) / 아닌 위치(False)

    주의:
        - DDP에서는 각 프로세스가 이미 로컬 배치만 받으므로 이 매핑이 필요 없음(그냥 반환 안 쓰면 됨).
        - DP 분할은 연속 청크 기준(start..end)로 가정.
    """
    if world_size <= 1:
        return (ct_img_idx, torch.ones_like(ct_img_idx, dtype=torch.bool)) if return_mask else ct_img_idx

    # per-shard 크기(불균등 분배 대응)
    base = global_batch_size // world_size
    rem  = global_batch_size % world_size
    local_size = base + (1 if device_id < rem else 0)
    start = device_id * base + min(device_id, rem)
    end   = start + local_size

    keep = (ct_img_idx >= start) & (ct_img_idx < end)
    if return_mask:
        return (ct_img_idx[keep] - start, keep)
    else:
        return ct_img_idx[keep] - start


def deterministic_scatter_add(src, index, out):
    """
    src: (Nc, H, W)
    index: (Nc, H, W), value in 0 ~ N-1
    out: (N, H, W)
    """
    Nc = src.shape[0]
    for i in range(Nc):
        out[index[i]] += src[i]


def deterministic_scatter_add_vectorized(src, index, dim_size):
    """
    결정론적으로 동작하는 벡터화된 scatter_add 함수.

    Args:
        src (Tensor): 소스 텐서. (N, D)
        index (Tensor): 인덱스 텐서. (N,)
        dim_size (int): 출력 텐서의 크기.

    Returns:
        Tensor: Scatter-add 연산이 완료된 결과 텐서. (dim_size, D)
    """
    # 1. 인덱스를 기준으로 정렬 순서를 찾습니다.
    perm = index.argsort()
    sorted_src = src[perm]
    sorted_index = index[perm]

    # 2. 그룹별 합산을 위해 각 그룹의 경계를 찾습니다.
    # sorted_index가 달라지는 지점이 그룹의 경계입니다.
    # [True]를 앞뒤로 붙여 첫 그룹과 마지막 그룹의 경계를 잡아줍니다.
    boundaries = torch.cat([
        torch.tensor([True], device=src.device),
        sorted_index[1:] != sorted_index[:-1],
        torch.tensor([True], device=src.device)
    ])

    # 3. 누적 합(cumulative sum)을 계산합니다.
    cumsum = torch.cumsum(sorted_src, dim=0)

    # 4. 경계 지점의 누적 합을 추출하여 그룹별 합산을 계산합니다.
    # (다음 경계의 누적 합) - (이전 경계의 누적 합) = (그룹의 합)
    segment_sums = cumsum[boundaries[1:]] - cumsum[boundaries[:-1]]

    # 5. 최종 결과를 담을 텐서를 만들고, 계산된 그룹별 합산을 뿌려줍니다.
    output = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    unique_indices = sorted_index[boundaries[:-1]]
    # 👇 unique_indices를 long 타입으로 변환하여 인덱싱 문제를 해결합니다.
    output[unique_indices.long()] = segment_sums

    return output


def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    ## debug
    for i in range(batch_size):
        # print(
        #     f"[DEBUG] i={i}, ct_01[i].shape={ct_01[i].shape}, ct_01[i].sum()={ct_01[i].sum()}, poly[i].shape={poly[i].shape}")
        assert ct_01[i].sum() <= poly[i].shape[0], f"Invalid index: ct_01[{i}] = {ct_01[i]}, poly[{i}].shape = {poly[i].shape}"

    for i in range(batch_size):
        valid = ct_01[i].sum().item()
        expected = poly[i].shape[0]
        # print(f"[DEBUG] i={i}, ct_01.sum={valid}, poly[i].shape[0]={expected}")
        assert valid <= expected, f"[ERROR] ct_01[{i}].sum()={valid} > poly[{i}].shape[0]={expected}"
    ##
    polys = []
    for i in range(batch_size):
        num = ct_01[i].sum()
        if num > 0:
            polys.append(poly[i][:num])
    if len(polys) > 0:
        poly = torch.cat(polys, dim=0)
    else:
        poly = torch.empty((0, poly.shape[2], poly.shape[3]), device=poly.device)
    # poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
    # poly = torch.cat([poly[i][:ct_01[i].sum()] for i in range(batch_size)], dim=0)
    return poly

def compute_winding_number(pys, point):
    '''
    Args:
        pys: (N_pys, N_pts, 2)
        point: (N_pys, N_pts, 2)

    Returns:

    '''
    pys = torch.unsqueeze(pys, -3).expand(-1, pys.size(1), -1, -1) # (N_pys, N_pts, N_pts, 2)
    pys_pre = torch.roll(pys, 1, -2)
    point = point.unsqueeze(-2)
    mask_add_winding = torch.logical_and(pys_pre[..., 1] <= point[..., 1], torch.logical_and(pys[..., 1] > point[..., 1], is_left(pys_pre, pys, point)))
    mask_minus_winding = torch.logical_and(pys_pre[..., 1] > point[..., 1], torch.logical_and(pys[..., 1] <= point[..., 1], is_left(pys, pys_pre, point)))
    winding_number = mask_add_winding.sum(-1) - mask_minus_winding.sum(-1)
    return winding_number

def is_left(pys1, pys2, point):
    return (pys2[...,0] - pys1[...,0]) * (point[...,1] - pys1[...,1]) - (point[...,0] - pys1[...,0]) * (pys2[...,1] - pys1[...,1]) > 0

def to_unit(vec):
    return vec / torch.norm(vec, dim=-1, keepdim=True).expand(-1, -1, vec.size(-1))


def get_normal_vec(pys_pre, pys, pys_nxt):
    # print(f"pys : {pys}")
    vec_to_next_compact = to_unit(pys_nxt - pys)
    vec_to_pre_compact = to_unit(pys_pre - pys)

    # ct = pys.mean(1) #N_pyx2

    normal_vec_next = torch.matmul(vec_to_next_compact, torch.tensor([[0, -1], [1, 0]], device=vec_to_next_compact.device).float()) #N_pyxN_ptsx2
    normal_vec_pre = torch.matmul(vec_to_pre_compact, torch.tensor([[0, 1], [-1, 0]], device=vec_to_pre_compact.device).float())

    # nxtedge_to_ct = ct.unsqueeze(1) - (pys_nxt + pys)/2 #N_pyxN_ptsx2
    # preedge_to_ct = ct.unsqueeze(1) - (pys_pre + pys)/2
    #
    # inner_with_ct_nxt = torch.matmul(nxtedge_to_ct.unsqueeze(2), normal_vec_next.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    # inner_with_ct_pre = torch.matmul(preedge_to_ct.unsqueeze(2), normal_vec_pre.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    #
    # vec_nxtct_to_prect = (pys_pre + pys) / 2 - (pys_nxt + pys) / 2 #N_pyxN_ptsx2
    # vec_prect_to_nxtct = (pys_nxt + pys) / 2 - (pys_pre + pys) / 2
    #
    # inner_with_prect_normal = torch.matmul(vec_nxtct_to_prect.unsqueeze(2), normal_vec_next.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    # inner_with_nxtct_normal = torch.matmul(vec_prect_to_nxtct.unsqueeze(2), normal_vec_pre.unsqueeze(-1)).squeeze(
    #     -1).squeeze(-1)
    #
    # mask_reverse_normal_nxt = torch.logical_or(inner_with_ct_nxt < 0, torch.logical_and(inner_with_ct_nxt == 0, inner_with_prect_normal < 0))
    # mask_reverse_normal_pre = torch.logical_or(inner_with_ct_pre < 0, torch.logical_and(inner_with_ct_pre == 0, inner_with_nxtct_normal < 0))

    # normal_vec_pre[mask_reverse_normal_pre.unsqueeze(-1).expand(-1,-1,2)] = (-1) * normal_vec_pre[mask_reverse_normal_pre.unsqueeze(-1).expand(-1,-1,2)]
    # normal_vec_next[mask_reverse_normal_nxt.unsqueeze(-1).expand(-1,-1,2)] = (-1) * normal_vec_next[mask_reverse_normal_nxt.unsqueeze(-1).expand(-1,-1,2)]

    normal_vec = (normal_vec_pre + normal_vec_next) / 2
    zero_normal_ids = torch.nonzero(torch.round(torch.norm(normal_vec, dim=-1, keepdim=True)) == 0)
    normal_vec_pre_rot90 = torch.matmul(normal_vec_pre, torch.tensor([[0, 1], [-1, 0]], device=vec_to_pre_compact.device).float())
    normal_vec[zero_normal_ids[:,0],zero_normal_ids[:,1],zero_normal_ids[:,2]] = normal_vec_pre_rot90[zero_normal_ids[:,0],zero_normal_ids[:,1],zero_normal_ids[:,2]]
    winding_numbers = compute_winding_number(pys, pys) # (N_pys, N_pts)
    mask_reverse = (winding_numbers > 0).unsqueeze(-1).expand(-1, -1, 2)
    normal_vec[mask_reverse] = (-1)*normal_vec[mask_reverse]
    # normal_vec = to_unit(normal_vec)

    return normal_vec, normal_vec_next, normal_vec_pre

#for deep snake
def get_adj_ind(n_adj, n_nodes, device):
    ind = torch.LongTensor([i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0])
    ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
    return ind.to(device)
def get_box_match_ind(pred_box, score, gt_poly, th_iou=0.7, th_confidence=0.1):
    if gt_poly.size(0) == 0:
        return [], []

    gt_box = torch.cat([torch.min(gt_poly, dim=1)[0], torch.max(gt_poly, dim=1)[0]], dim=1)
    iou_matrix = data_utils.box_iou(pred_box, gt_box)  # (n, m)
    iou, gt_ind = iou_matrix.max(dim=1)
    box_ind = ((iou > th_iou) * (score > th_confidence)).nonzero().view(-1)
    gt_ind = gt_ind[box_ind] # filtering for gt_ind

    ind = np.unique(gt_ind.detach().cpu().numpy(), return_index=True)[1] #not allowed duplicate (but obtain only first occurence)
    box_ind = box_ind[ind]
    gt_ind = gt_ind[ind]

    return box_ind, gt_ind

def prepare_training_init(ret, batch, ro, num_points=128, with_img_idx=False):
    # ct_01 = batch['ct_01'].byte()
    ct_01 = batch['ct_01'].bool()
    init = {}
    init.update({'img_it_init_polys': collect_training(batch['img_it_init_polys'], ct_01)})
    init.update({'can_it_init_polys': collect_training(batch['can_it_init_polys'], ct_01)})
    init.update({'img_gt_init_polys': collect_training(batch['img_gt_init_polys'], ct_01)})
    init.update({'can_gt_init_polys': collect_training(batch['can_gt_init_polys'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'ct_01': ct_01})
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init

def prepare_training(ret, batch, ro=4., num_points=128, with_img_idx=False, init_key='poly_coarse', set_py=True):
    # ct_01 = batch['ct_01'].byte()
    ct_01 = batch['ct_01'].bool()
    init = {}
    init.update({'img_gt_coarse_polys': collect_training(batch['img_gt_coarse_polys'], ct_01)})
    init.update({'img_gt_init_polys': collect_training(batch['img_gt_init_polys'], ct_01)})
    init.update({'img_gt_polys': collect_training(batch['img_gt_polys'], ct_01)})
    if set_py:
        if ret[init_key].size(-2) != num_points:
            if with_img_idx:
                poly_coarse = uniform_upsample(ret[init_key].detach(), num_points).squeeze(0)
            else:
                poly_coarse = uniform_upsample(ret[init_key].detach().unsqueeze(0), num_points).squeeze(0)
        else:
            poly_coarse = ret[init_key].detach()
        init.update({'img_init_polys': poly_coarse / ro})
        can_init_polys = img_poly_to_can_poly(poly_coarse / ro)
        init.update({'can_init_polys': can_init_polys})

    ct_num = batch['meta']['ct_num']
    init.update({'py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'ct_01': ct_01})
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init

def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    return can_poly

# def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
#     img_poly = img_poly.clone()
#     img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
#     img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
#
#     batch_size = cnn_feature.size(0)
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
#         ind: [N_poly] — 이미지 인덱스 (0 ~ B-1)
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
    # 🔍 index 검사
    assert (ind >= 0).all(), f"ind has negative index: {ind} (evolve)"
    assert (ind < B).all(), f"ind has index out of range: max={ind.max().item()}, B={B} (evolve)"
    ##
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

def get_gcn_feature_3x3(cnn_feature, img_poly, ind, h, w, mode='flatten', detach_features=False):
    """
    Extract 3x3 patch features around each vertex position.
    
    Args:
        cnn_feature: [B, C, H, W]
        img_poly: [N_poly, N_vert, 2] (pixel coords)
        ind: [N_poly] (image indices for each poly)
        h, w: height and width of feature map
        mode: 'flatten' or 'spatial'
            - 'flatten': returns [N_poly, C*9, N_vert]
            - 'spatial': returns [N_poly, C, N_vert, 3, 3]
        detach_features: True to detach features (no gradient), False to keep gradient
    
    Returns:
        gcn_feature: extracted 3x3 features
    """
    B, C = cnn_feature.shape[:2]
    N_poly = img_poly.size(0)
    N_vert = img_poly.size(1)
    
    if N_poly == 0:
        if mode == 'flatten':
            return torch.empty(0, C*9, 1, device=cnn_feature.device)
        else:
            return torch.empty(0, C, 1, 3, 3, device=cnn_feature.device)
    
    # Define 3x3 offsets in normalized coordinates
    # Feature map에서 1픽셀 = 2/w or 2/h in normalized coords
    offset_x = 2.0 / w  # normalized offset for 1 pixel in x
    offset_y = 2.0 / h  # normalized offset for 1 pixel in y
    
    # Create 3x3 offset grid
    offsets = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            offsets.append([dx * offset_x, dy * offset_y])
    
    # Normalize polygon coordinates to [-1, 1]
    img_poly_norm = img_poly.clone()
    img_poly_norm[..., 0] = img_poly_norm[..., 0] / (w / 2.) - 1
    img_poly_norm[..., 1] = img_poly_norm[..., 1] / (h / 2.) - 1
    
    # Prepare output tensor based on mode
    if mode == 'flatten':
        gcn_feature = torch.zeros(N_poly, C * 9, N_vert, device=img_poly.device, dtype=cnn_feature.dtype)
    else:  # spatial mode
        gcn_feature = torch.zeros(N_poly, C, N_vert, 3, 3, device=img_poly.device, dtype=cnn_feature.dtype)
    
    # Extract features for each image in batch
    for i in range(B):
        mask = (ind == i)
        if mask.sum() == 0:
            continue
            
        polys_i = img_poly_norm[mask]  # [N_poly_i, N_vert, 2]
        features_list = []
        
        # Sample each offset position
        for offset_idx, (dx, dy) in enumerate(offsets):
            poly_offset = polys_i.clone()
            poly_offset[..., 0] += dx
            poly_offset[..., 1] += dy
            
            # Clamp to valid range [-1, 1]
            poly_offset[..., 0] = torch.clamp(poly_offset[..., 0], -1, 1)
            poly_offset[..., 1] = torch.clamp(poly_offset[..., 1], -1, 1)
            
            # Sample features: [1, N_poly_i, N_vert, 2]
            poly_offset = poly_offset.unsqueeze(0).permute(0, 2, 1, 3)  # -> [1, N_vert, N_poly_i, 2]
            sampled = F.grid_sample(cnn_feature[i:i+1], poly_offset, align_corners=True)
            feat = sampled.squeeze(0).permute(2, 0, 1)  # -> [N_poly_i, C, N_vert]
            
            # Apply detach option for gradient control
            if detach_features:
                features_list.append(feat.clone().detach())
            else:
                features_list.append(feat)
        
        # Use cat instead of stack+permute+reshape to avoid view issues
        if mode == 'flatten':
            # Direct concatenation: [N_poly_i, C, N_vert] * 9 -> [N_poly_i, C*9, N_vert]
            features_3x3 = torch.cat(features_list, dim=1)  # [N_poly_i, C*9, N_vert]
            gcn_feature[mask] = features_3x3
        else:
            # For spatial mode, still need stack+view
            features_3x3 = torch.stack(features_list, dim=-1)  # [N_poly_i, C, N_vert, 9]
            features_3x3 = features_3x3.view(mask.sum(), C, N_vert, 3, 3)
            gcn_feature[mask] = features_3x3
    
    return gcn_feature

def get_gcn_feature_window(cnn_feature, img_poly, ind, h, w, window_stride, window_size, grad_neighbors=True):
    window_list = []

    c_id = 0
    for w_h in range(-int((window_size[0]-1)/2), int((window_size[0]-1)/2)+1):
        for w_w in range(-int((window_size[1]-1)/2), int((window_size[1]-1)/2)+1):
            window_list.append([w_w,w_h])
            if w_w == 0 and w_h == 0:
                center_id = c_id
            c_id += 1

    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    # periphery : or it can be conducted here?

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1), window_size[0], window_size[1]]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        # print(cnn_feature[i:i+1].shape)
        # print(poly.shape)
        # periphery
        feature_append_list = []
        for w_id, w in enumerate(window_list):
            poly_tmp = poly.clone()
            poly_tmp[..., 0] = poly_tmp[..., 0] + (window_stride * w[0])
            poly_tmp[..., 1] = poly_tmp[..., 1] + (window_stride * w[1])
            # print(f"{poly_append.shape}&{poly_tmp.append}")
            feature = torch.nn.functional.grid_sample(cnn_feature[i:i + 1], poly_tmp)[0].permute(1, 0, 2)
            if (not grad_neighbors) and (w_id != center_id):
                print(f"WARNING !! grad_neighbors is {grad_neighbors} AND w_id : {w_id} (!= center_id {center_id})")
                feature = feature.clone().detach()
            feature_append_list.append(feature)
            del poly_tmp, feature

        feature_append = torch.stack(feature_append_list, -1)
        feature_append = feature_append.contiguous().view(feature_append.size(0), feature_append.size(1), feature_append.size(2), window_size[0], window_size[1])

        gcn_feature[ind == i] = feature_append
        del feature_append_list, poly, feature_append

    del window_list
    return gcn_feature

def prepare_testing_init(polys, ro, num_points=128, py_ind=None, set_py=True):
    if set_py:
        if polys.size(-2) != num_points:
            polys_upsample = uniform_upsample(polys.unsqueeze(0), num_points).squeeze(0)
        else:
            polys_upsample = polys
        polys_upsample = polys_upsample / ro
        can_init_polys = img_poly_to_can_poly(polys_upsample)
        img_init_polys = polys_upsample
        init = {'img_init_polys': img_init_polys, 'can_init_polys': can_init_polys}
    else:
        init = {}
    if py_ind is not None:
        init['py_ind'] = py_ind
    return init


# def uniform_upsample(poly, p_num):
#     from network.detector_decode.extreme_utils import _ext as extreme_utils
#     # 1. assign point number for each edge
#     # 2. calculate the coefficient for linear interpolation
#     next_poly = torch.roll(poly, -1, 2)
#     edge_len = (next_poly - poly).pow(2).sum(-1).sqrt()
#     edge_num = torch.round(edge_len * p_num / torch.sum(edge_len, dim=-1)[..., None]).long()
#     edge_num = torch.clamp(edge_num, min=1)
#     edge_num_sum = torch.sum(edge_num, dim=-1)
#     edge_idx_sort = torch.argsort(edge_num, dim=-1, descending=True)
#     extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
#     edge_num_sum = torch.sum(edge_num, dim=-1)
#     assert torch.all(edge_num_sum == p_num)
#
#     edge_start_idx = torch.cumsum(edge_num, dim=-1) - edge_num
#     weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
#     poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
#     poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
#     poly = poly1 * (1 - weight) + poly2 * weight
#
#     return poly