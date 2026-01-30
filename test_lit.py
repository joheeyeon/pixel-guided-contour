import os, argparse, importlib, tqdm, torch, time, random, numpy as np, cv2
import pytorch_lightning as pl
from network import make_network
from dataset.data_loader import make_data_loader
from dataset.collate_batch import collate_batch
from train.model_utils.utils import load_network_lightning
from evaluator.make_evaluator import make_evaluator
from train.trainer.utils import SegmentationMetrics
from scipy.io import savemat
import nms
import torch.nn.functional as F
import math
try:
    from self_intersection_viz_corrected import visualize_self_intersections_corrected
    use_corrected_viz = True
except ImportError:
    use_corrected_viz = False
    try:
        from self_intersection_viz_fixed import visualize_self_intersections_fixed
        visualize_self_intersections_improved = visualize_self_intersections_fixed
    except ImportError:
        try:
            from self_intersection_viz_improved import visualize_self_intersections_improved
        except ImportError:
            from self_intersection_viz import visualize_self_intersections
            visualize_self_intersections_improved = visualize_self_intersections

def _is_multi_process():
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1

# ========================================================================
# Snake Energy Functions
# ========================================================================

def gaussian_blur(img, k=9, sigma=2.0):
    # separable 1D Gaussian
    half = k//2
    x = torch.arange(-half, half+1, device=img.device, dtype=img.dtype)
    g = torch.exp(-0.5*(x/sigma)**2); g = g/g.sum()
    img = F.conv2d(img, g.view(1,1,1,k), padding=(0,half))
    img = F.conv2d(img, g.view(1,1,k,1), padding=(half,0))
    return img

def gradient_magnitude(img):
    # simple Sobel
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    gx = F.conv2d(img, kx, padding=1)
    gy = F.conv2d(img, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)

def sample_bilinear(field, xy):  # xy: N x 2 in pixel coords (x,y)
    # grid_sample expects coords in [-1,1]
    N, _ = xy.shape
    H, W = field.shape[-2:]
    gx = (xy[:,0] / (W-1)) * 2 - 1
    gy = (xy[:,1] / (H-1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, N, 1, 2)
    vals = F.grid_sample(field, grid, align_corners=True).view(N)
    return vals

def snake_energy(image_gray, contour_xy, alpha=0.1, beta=1.0, gamma=1.0, sigma=2.0, normalize=True):
    """
    image_gray: torch.Tensor [H,W] or [1,1,H,W], values in [0,1]
    contour_xy: torch.Tensor [N,2] as (x,y) pixel coords, closed polygon assumed
    returns dict with E_cont, E_curv, E_ext, E_total
    """
    if image_gray.dim()==2:
        img = image_gray[None,None]
    elif image_gray.shape[0]==1 and image_gray.shape[1]==1:
        img = image_gray
    else:
        raise ValueError("image_gray must be [H,W] or [1,1,H,W]")

    # external potential P = -|∇(Gσ * I)|
    sm = gaussian_blur(img, k=max(3, int(6*sigma)|1), sigma=sigma)
    gradmag = gradient_magnitude(sm)
    P = -gradmag  # higher (more negative) near edges

    v = contour_xy  # [N,2]
    N = v.shape[0]
    if N < 3:  # too few points
        return {"E_cont": 0.0, "E_curv": 0.0, "E_ext": 0.0, "E_total": 0.0}
    
    idxp = torch.roll(torch.arange(N), -1)
    idxm = torch.roll(torch.arange(N), 1)

    # finite differences (closed)
    d1 = v[idxp] - v            # first difference (continuity)
    d2 = v[idxp] - 2*v + v[idxm]# second difference (curvature)

    # optional scale normalization by mean segment length
    if normalize:
        seg_len = torch.sqrt((d1**2).sum(dim=1) + 1e-12)
        Lbar = seg_len.mean().clamp(min=1.0)
        d1 = d1 / Lbar
        d2 = d2 / (Lbar**2)

    E_cont = (d1.pow(2).sum(dim=1)).mean() * alpha
    E_curv = (d2.pow(2).sum(dim=1)).mean() * beta

    # sample P at contour vertices (bilinear)
    P_vals = sample_bilinear(P, v)
    E_ext  = P_vals.mean() * gamma

    E_total = E_cont + E_curv + E_ext
    return {
        "E_cont": E_cont.item(),
        "E_curv": E_curv.item(),
        "E_ext":  E_ext.item(),
        "E_total": E_total.item()
    }

def format_energy_text(energy_dict):
    """Format energy values for display (4 digits total)"""
    e_total = energy_dict["E_total"]
    if abs(e_total) < 0.01:
        return f"{e_total:.3f}"
    elif abs(e_total) < 0.1:
        return f"{e_total:.3f}"
    elif abs(e_total) < 1.0:
        return f"{e_total:.3f}"
    elif abs(e_total) < 10.0:
        return f"{e_total:.2f}"
    elif abs(e_total) < 100.0:
        return f"{e_total:.1f}"
    else:
        return f"{e_total:.0f}"

# ========================================================================

def _to_uint8(img_chw):
    # img: (C,H,W), float/torch -> uint8 BGR
    if isinstance(img_chw, torch.Tensor):
        img = img_chw.detach().cpu().float().numpy()
    else:
        img = img_chw.astype(np.float32)
    img = np.transpose(img, (1, 2, 0))  # HWC
    # 대충 0~1 또는 -2~2 같은 정규화 대응: 0~255로 스케일링
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        img = np.zeros_like(img)
    else:
        img = (img - mn) / (mx - mn)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    # 모델 입력이 RGB일 가능성 → OpenCV는 BGR이므로 변환 없이 그려도 크게 무방
    return img

def _draw_polys(img, polys, color, thickness=2):
    """ polys: list[Tensor/ndarray (Nv,2)] 또는 Tensor(list-like) """
    out = img.copy()
    if polys is None:
        return out
    # 배치 단위일 수 있으니 i번째만 들어오도록 상단 호출부에서 주의
    for poly in polys:
        if poly is None:
            continue
        if isinstance(poly, torch.Tensor):
            pts = poly.detach().cpu().numpy()
        else:
            pts = np.asarray(poly)
        if pts.size == 0:
            continue
        pts = np.round(pts).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out

def _draw_polys_with_energy(img, polys, color, thickness=2, image_gray=None, show_energy=False):
    """
    Draw polygons with optional snake energy values displayed as text
    img: BGR image to draw on
    polys: list of polygons
    color: drawing color
    thickness: line thickness
    image_gray: grayscale image for energy calculation [H,W] tensor or numpy
    show_energy: whether to calculate and display energy values
    """
    out = img.copy()
    if polys is None:
        return out
    
    # Convert image to tensor if needed for energy calculation
    if show_energy and image_gray is not None:
        if isinstance(image_gray, np.ndarray):
            image_tensor = torch.from_numpy(image_gray).float()
            if image_tensor.max() > 1.0:  # normalize to [0,1] if needed
                image_tensor = image_tensor / 255.0
        else:
            image_tensor = image_gray.float()
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
    
    for i, poly in enumerate(polys):
        if poly is None:
            continue
        if isinstance(poly, torch.Tensor):
            pts = poly.detach().cpu().numpy()
            poly_tensor = poly.detach().cpu().float()
        else:
            pts = np.asarray(poly)
            poly_tensor = torch.from_numpy(pts).float()
        
        if pts.size == 0:
            continue
        
        pts = np.round(pts).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
        
        # Calculate and display energy if requested
        if show_energy and image_gray is not None and poly_tensor.shape[0] >= 3:
            try:
                energy_dict = snake_energy(image_tensor, poly_tensor)
                energy_text = format_energy_text(energy_dict)
                
                # Find centroid for text placement
                centroid_x = int(pts[:, 0].mean())
                centroid_y = int(pts[:, 1].mean())
                
                # Add background rectangle for better readability
                text_size = cv2.getTextSize(energy_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(out, 
                            (centroid_x - text_size[0]//2 - 2, centroid_y - text_size[1] - 2),
                            (centroid_x + text_size[0]//2 + 2, centroid_y + 2),
                            (0, 0, 0), -1)  # black background
                
                # Add text
                cv2.putText(out, energy_text, 
                          (centroid_x - text_size[0]//2, centroid_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # white text
            except Exception as e:
                # Skip energy calculation if failed
                pass
    
    return out

def visualize_gt(batch, save_dir="debug_gt"):
    """
    왼쪽: init (batch['img_gt_polys'])
    오른쪽: coarse (batch['img_gt_coarse_polys'])
    파일명: batch['meta']['img_name'][i] 사용
    """
    os.makedirs(save_dir, exist_ok=True)

    imgs = batch['inp']                 # (B,C,H,W)
    B = imgs.shape[0]

    # 이미지 이름 확보
    img_names = None
    if ('meta' in batch) and isinstance(batch['meta'], dict) and ('img_name' in batch['meta']):
        img_names = batch['meta']['img_name']
        # img_names가 tensor가 아니라 list[str] 형태라고 가정. 아니면 변환
        if isinstance(img_names, torch.Tensor):
            img_names = [str(x) for x in img_names]

    gt_init_all = batch.get('img_gt_polys', None)
    gt_coarse_all = batch.get('img_gt_coarse_polys', None)

    # 배치 구조가 리스트-오브-리스트(각 이미지별 다수 폴리곤)라고 가정
    for i in range(B):
        base = _to_uint8(imgs[i])  # H,W,C (uint8)

        # 왼쪽: init (초록)
        left = _draw_polys(base, None if gt_init_all is None else gt_init_all[i]*4, (0,255,0), 2)

        # 오른쪽: coarse (빨강)
        right = _draw_polys(base, None if gt_coarse_all is None else gt_coarse_all[i]*4, (0,0,255), 2)

        # 가로로 붙이기 + 가운데 하얀 구분선
        h, w, _ = left.shape
        sep = np.full((h, 4, 3), 255, dtype=np.uint8)  # 얇은 흰색 벽
        vis = np.concatenate([left, sep, right], axis=1)

        # 파일명 정하기
        if img_names and i < len(img_names):
            stem = os.path.splitext(os.path.basename(img_names[i]))[0]
            save_path = os.path.join(save_dir, f"{stem}.png")
        else:
            save_path = os.path.join(save_dir, f"{i:04d}.png")

        cv2.imwrite(save_path, vis)

def visualize_pixel_maps(batch, output, save_dir="pixel_viz", batch_idx=0, mode='final', cfg=None):
    """
    Pixel map 시각화
    Args:
        batch: 입력 배치 데이터
        output: 네트워크 출력
        save_dir: 저장 디렉토리
        batch_idx: 배치 인덱스
        mode: 'final' (최종 pixel map만) 또는 'all' (모든 단계별 pixel map)
        cfg: 설정 객체 (ccp_deform_pixel_norm, viz_pixel_raw 확인용)
    """
    import os
    import numpy as np
    import cv2
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 이미지 가져오기
    imgs = batch['inp']  # (B, C, H, W)
    B = imgs.shape[0]
    
    # 이미지 이름 확보
    img_names = None
    if ('meta' in batch) and isinstance(batch['meta'], dict) and ('img_name' in batch['meta']):
        img_names = batch['meta']['img_name']
        if isinstance(img_names, torch.Tensor):
            img_names = [str(x) for x in img_names]
    
    # Pixel maps 추출
    if 'pixel' not in output:
        print("[WARN] No pixel maps in output")
        return
    
    pixel_maps = output['pixel']
    if not isinstance(pixel_maps, list):
        pixel_maps = [pixel_maps]
    
    # raw 시각화 옵션 확인
    viz_raw = (cfg is not None and getattr(cfg.test, 'viz_pixel_raw', False))
    
    # ccp_deform_pixel_norm 확인 (기존 unnormalized 설정)
    is_unnormalized = (cfg is not None and 
                      hasattr(cfg.model, 'ccp_deform_pixel_norm') and 
                      cfg.model.ccp_deform_pixel_norm == 'unnormalized')
    
    # 각 배치 이미지에 대해
    for i in range(B):
        base_img = _to_uint8(imgs[i])  # (H, W, C)
        H, W = base_img.shape[:2]
        
        if mode == 'final':
            # 최종 pixel map만 시각화
            pixel_map = pixel_maps[-1][i] if len(pixel_maps) > 0 else None
            if pixel_map is None:
                continue
                
            pixel_map = pixel_map.detach().cpu()
            
            if viz_raw and pixel_map.shape[0] > 1:
                # --viz_pixel_raw 옵션: 항상 unnormalized 형태로 시각화
                _visualize_unnormalized_channels(pixel_map, base_img, save_dir, 
                                               img_names, i, batch_idx, stage_name="final_raw")
                continue
            elif is_unnormalized and pixel_map.shape[0] > 1:
                # 기존 unnormalized 설정일 때 각 채널별로 시각화
                _visualize_unnormalized_channels(pixel_map, base_img, save_dir, 
                                               img_names, i, batch_idx, stage_name="final")
                continue
            else:
                # 기존 방식 (normalized)
                if pixel_map.shape[0] > 1:
                    pixel_prob = torch.softmax(pixel_map, dim=0)[1]
                else:
                    pixel_prob = torch.sigmoid(pixel_map[0])
                
                # 원본 크기로 리사이즈
                pixel_resized = torch.nn.functional.interpolate(
                    pixel_prob.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode='nearest'
                ).squeeze().numpy()
                
                # 컬러맵 적용
                pixel_colored = (pixel_resized * 255).astype(np.uint8)
                pixel_colored = cv2.applyColorMap(pixel_colored, cv2.COLORMAP_JET)
                
                # 원본 이미지와 오버레이
                overlay = cv2.addWeighted(base_img, 0.5, pixel_colored, 0.5, 0)
                
                # 파일 저장
                if img_names and i < len(img_names):
                    stem = os.path.splitext(os.path.basename(img_names[i]))[0]
                    save_path = os.path.join(save_dir, f"{stem}_b{batch_idx:04d}.png")
                else:
                    save_path = os.path.join(save_dir, f"batch{batch_idx:04d}_img{i:04d}.png")
                
                cv2.imwrite(save_path, overlay)
            
        elif mode == 'all':
            if viz_raw:
                # --viz_pixel_raw 옵션: 모든 stage에서 unnormalized 형태로 시각화
                for stage_idx, pixel_map in enumerate(pixel_maps):
                    if pixel_map is None:
                        continue
                    pixel_map_i = pixel_map[i].detach().cpu()
                    if pixel_map_i.shape[0] > 1:
                        _visualize_unnormalized_channels(pixel_map_i, base_img, save_dir,
                                                       img_names, i, batch_idx, 
                                                       stage_name=f"stage{stage_idx}_raw")
                continue
            elif is_unnormalized:
                # 기존 unnormalized 설정일 때 각 stage별로 채널 분리 시각화
                for stage_idx, pixel_map in enumerate(pixel_maps):
                    if pixel_map is None:
                        continue
                    pixel_map_i = pixel_map[i].detach().cpu()
                    if pixel_map_i.shape[0] > 1:
                        _visualize_unnormalized_channels(pixel_map_i, base_img, save_dir,
                                                       img_names, i, batch_idx, 
                                                       stage_name=f"stage{stage_idx}")
                continue
            else:
                # 기존 방식: 모든 단계별 pixel map을 가로로 나열
                stages = []
                
                for stage_idx, pixel_map in enumerate(pixel_maps):
                    if pixel_map is None:
                        continue
                        
                    pixel_map_i = pixel_map[i].detach().cpu()
                    if pixel_map_i.shape[0] > 1:
                        pixel_prob = torch.softmax(pixel_map_i, dim=0)[1]
                    else:
                        pixel_prob = torch.sigmoid(pixel_map_i[0])
                    
                    # 원본 크기로 리사이즈
                    pixel_resized = torch.nn.functional.interpolate(
                        pixel_prob.unsqueeze(0).unsqueeze(0),
                        size=(H, W), mode='nearest'
                    ).squeeze().numpy()
                    
                    # 컬러맵 적용
                    pixel_colored = (pixel_resized * 255).astype(np.uint8)
                    pixel_colored = cv2.applyColorMap(pixel_colored, cv2.COLORMAP_JET)
                    
                    # 원본 이미지와 오버레이
                    stage_overlay = cv2.addWeighted(base_img, 0.5, pixel_colored, 0.5, 0)
                    stages.append(stage_overlay)
                
                if len(stages) == 0:
                    continue
                
                # 흰색 구분선 추가하여 가로로 연결
                sep = np.full((H, 10, 3), 255, dtype=np.uint8)  # 10픽셀 폭의 흰색 구분선
                overlay = stages[0]
                for stage_img in stages[1:]:
                    overlay = np.concatenate([overlay, sep, stage_img], axis=1)
                
                # 파일 저장 (all 모드)
                if img_names and i < len(img_names):
                    stem = os.path.splitext(os.path.basename(img_names[i]))[0]
                    save_path = os.path.join(save_dir, f"{stem}_b{batch_idx:04d}.png")
                else:
                    save_path = os.path.join(save_dir, f"batch{batch_idx:04d}_img{i:04d}.png")
                
                cv2.imwrite(save_path, overlay)


def _visualize_unnormalized_channels(pixel_map, base_img, save_dir, img_names, img_idx, batch_idx, stage_name="final"):
    """
    unnormalized pixel map의 각 채널을 분리하여 colorbar와 함께 시각화
    Args:
        pixel_map: unnormalized pixel tensor (C, H_feat, W_feat)
        base_img: 원본 이미지 (H, W, 3)
        save_dir: 저장 디렉토리
        img_names: 이미지 이름 리스트
        img_idx: 배치 내 이미지 인덱스
        batch_idx: 배치 인덱스
        stage_name: 스테이지 이름
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    H, W = base_img.shape[:2]
    C = pixel_map.shape[0]
    
    # 파일명 생성
    if img_names and img_idx < len(img_names):
        stem = os.path.splitext(os.path.basename(img_names[img_idx]))[0]
        base_filename = f"{stem}_b{batch_idx:04d}_{stage_name}"
    else:
        base_filename = f"batch{batch_idx:04d}_img{img_idx:04d}_{stage_name}"
    
    # Skip individual channel visualization to save storage space
    # Only create combined visualization below
    
    # 모든 채널을 하나의 그림에 표시 (선택적)
    if C > 1:
        fig, axes = plt.subplots(1, C + 1, figsize=(5 * (C + 1), 5))
        
        # 첫 번째: 원본 이미지
        axes[0].imshow(base_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 나머지: 각 채널
        for c in range(C):
            channel_map = pixel_map[c]
            channel_resized = torch.nn.functional.interpolate(
                channel_map.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='nearest'
            ).squeeze().numpy()
            
            vmin, vmax = channel_resized.min(), channel_resized.max()
            im = axes[c + 1].imshow(channel_resized, cmap='jet', vmin=vmin, vmax=vmax)
            
            channel_name = 'Background' if c == 0 else 'Foreground'
            axes[c + 1].set_title(f'Ch{c} ({channel_name})\n[{vmin:.3f}, {vmax:.3f}]')
            axes[c + 1].axis('off')
            
            # 작은 colorbar
            cbar = plt.colorbar(im, ax=axes[c + 1], shrink=0.6)
            cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{base_filename}_all_channels.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def visualize_stage1_pixel_results(batch, output, save_dir="stage1_viz", batch_idx=0):
    """
    Stage 1 전용 시각화: 원본 이미지 + pixel mask + GT mask 오버레이
    """
    if 'pixel' not in output:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    # 입력 이미지 및 예측 pixel map
    imgs = batch['inp']  # (B,C,H,W)
    pix = output['pixel'][-1] if isinstance(output['pixel'], list) else output['pixel']
    
    B = imgs.shape[0]
    
    for b_idx in range(B):
        try:
            # 원본 이미지 (RGB로 변환)
            img = _to_uint8(imgs[b_idx])  # (H,W,3)
            H, W = img.shape[:2]
            
            # Pixel prediction 처리
            if pix.shape[1] == 1:
                # Binary classification
                prob = torch.sigmoid(pix[b_idx, 0]).cpu().numpy()  # (H,W)
                th = float(getattr(batch.get('cfg', type('', (), {'test': type('', (), {'pixel_th': 0.5})()})), 'test', type('', (), {'pixel_th': 0.5})).pixel_th)
                pred_mask = (prob >= th).astype(np.uint8) * 255
            else:
                # Multi-class classification
                pred_logits = pix[b_idx].cpu().numpy()  # (C,H,W)
                pred_class = np.argmax(pred_logits, axis=0)  # (H,W)
                pred_mask = (pred_class != 0).astype(np.uint8) * 255  # foreground mask
            
            # GT mask 처리
            gt_mask = None
            if 'pixel_gt' in batch:
                gt_pixel = batch['pixel_gt'][b_idx].cpu().numpy()  # (H,W)
                if gt_pixel.shape != (H, W):
                    gt_pixel = cv2.resize(gt_pixel.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                gt_mask = (gt_pixel > 0).astype(np.uint8) * 255
            
            # 시각화 생성
            # 1) 원본 이미지
            vis_orig = img.copy()
            
            # 2) 예측 마스크 오버레이 (빨간색)
            vis_pred = img.copy()
            if pred_mask.shape != (H, W):
                pred_mask = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            pred_colored = np.zeros_like(img)
            pred_colored[pred_mask > 0] = [0, 0, 255]  # 빨간색
            vis_pred = cv2.addWeighted(vis_pred, 0.7, pred_colored, 0.3, 0)
            
            # 3) GT 마스크 오버레이 (초록색) - GT가 있는 경우에만
            vis_gt = img.copy()
            if gt_mask is not None:
                if gt_mask.shape != (H, W):
                    gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                gt_colored = np.zeros_like(img)
                gt_colored[gt_mask > 0] = [0, 255, 0]  # 초록색
                vis_gt = cv2.addWeighted(vis_gt, 0.7, gt_colored, 0.3, 0)
            
            # 4) 비교 시각화 (예측 vs GT)
            if gt_mask is not None:
                vis_compare = img.copy()
                # TP: 노란색, FP: 빨간색, FN: 초록색
                tp_mask = (pred_mask > 0) & (gt_mask > 0)
                fp_mask = (pred_mask > 0) & (gt_mask == 0)
                fn_mask = (pred_mask == 0) & (gt_mask > 0)
                
                compare_colored = np.zeros_like(img)
                compare_colored[tp_mask] = [0, 255, 255]  # 노란색 (TP)
                compare_colored[fp_mask] = [0, 0, 255]    # 빨간색 (FP)
                compare_colored[fn_mask] = [0, 255, 0]    # 초록색 (FN)
                vis_compare = cv2.addWeighted(vis_compare, 0.7, compare_colored, 0.3, 0)
            else:
                vis_compare = vis_pred
            
            # 파일명 생성
            img_name = "unknown"
            if 'meta' in batch and 'img_name' in batch['meta'] and b_idx < len(batch['meta']['img_name']):
                img_name = os.path.splitext(os.path.basename(batch['meta']['img_name'][b_idx]))[0]
            
            # Binary mask만 저장 (단순 버전)
            save_path = os.path.join(save_dir, f"{img_name}_stage1_pixel.png")
            cv2.imwrite(save_path, pred_mask)
            
            # 아래 행만 저장 (GT와 비교)
            if gt_mask is not None:
                bottom_row = np.hstack([vis_gt, vis_compare])
                # 텍스트 라벨 추가
                cv2.putText(bottom_row, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(bottom_row, "Comparison", (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                save_path_compare = os.path.join(save_dir, f"{img_name}_stage1_compare.png")
                cv2.imwrite(save_path_compare, bottom_row)
            
        except Exception as e:
            print(f"[WARN] Stage 1 visualization failed for image {b_idx}: {e}")
            continue

def visualize_pixel_initial_contours(batch, output, save_dir="pixel_initial_contours_viz", batch_idx=0):
    """
    pixel map에서 추출한 ct_score 필터링 전 initial contour 시각화
    """
    if 'pixel_initial_contours' not in output:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    pixel_initial_contours = output['pixel_initial_contours']  # dict: {batch_idx: [contours]}
    
    # 색상 팔레트
    colors = [
        (0, 255, 0),    # 초록
        (255, 0, 0),    # 빨강
        (0, 0, 255),    # 파랑
        (255, 255, 0),  # 노랑
        (255, 0, 255),  # 자홍
        (0, 255, 255),  # 청록
        (128, 0, 255),  # 보라
        (255, 128, 0),  # 주황
        (0, 128, 255),  # 하늘
        (128, 255, 0),  # 연두
    ]
    
    # 이미지를 unnormalize (정규화 해제)
    def unnormalize_to_bgr(img_tensor):
        # config에서 mean, std 가져오기
        mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(3, 1, 1)
        
        # 텐서를 numpy로 변환 후 정규화 해제
        img_np = img_tensor.detach().cpu().numpy()  # (C, H, W)
        unnormalized_img = img_np * std + mean  # 정규화 해제
        img_np = unnormalized_img.transpose(1, 2, 0)  # (H, W, C)
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)  # 0-255 범위로 스케일링
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # OpenCV용 BGR 변환
    
    batch_size = batch['inp'].size(0)
    img_names = batch['meta']['img_name'] if 'meta' in batch else ['unknown'] * batch_size
    
    # 배치의 모든 이미지에 대해 처리
    for b_idx in range(batch_size):
        if b_idx not in pixel_initial_contours:
            continue
            
        try:
            base_img = unnormalize_to_bgr(batch['inp'][b_idx])
            H, W = base_img.shape[:2]
            
            contours = pixel_initial_contours[b_idx]  # list of (Nv,2) arrays
            
            if len(contours) == 0:
                continue
            
            # 원본 이미지 위에 contour 그리기
            viz_img = base_img.copy()
            
            for i, contour in enumerate(contours):
                color = colors[i % len(colors)]
                
                # contour는 (Nv, 2) numpy array: [x, y] 좌표
                if len(contour) >= 3:  # 최소 3점 이상
                    # down_ratio 적용 (필터링 전이므로 네트워크 내부 좌표)
                    contour_scaled = contour * 4  # down_ratio=4 적용
                    contour_int = np.round(contour_scaled).astype(np.int32)
                    
                    # 화면 범위 내로 클리핑
                    contour_int[:, 0] = np.clip(contour_int[:, 0], 0, W-1)
                    contour_int[:, 1] = np.clip(contour_int[:, 1], 0, H-1)
                    
                    # 폴리곤 그리기
                    cv2.polylines(viz_img, [contour_int], isClosed=True, color=color, thickness=2)
                    
                    # 컨투어 번호 표시
                    cx = int(contour_int[:, 0].mean())
                    cy = int(contour_int[:, 1].mean())
                    cv2.putText(viz_img, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 저장
            img_name = os.path.basename(str(img_names[b_idx]))
            stem = os.path.splitext(img_name)[0]
            save_path = os.path.join(save_dir, f"{stem}_b{batch_idx:04d}_i{b_idx:02d}_pixel_initial.png")
            cv2.imwrite(save_path, viz_img)
            
            print(f"[PIXEL INITIAL] Saved {len(contours)} contours for {img_name}")
            
        except Exception as e:
            print(f"[WARN] Pixel initial contour visualization failed for image {b_idx}: {e}")
            continue

def visualize_stage1_poly_init(batch, output, save_dir="init"):
    """
    Stage 1에서 ct_hm 또는 wh head가 학습될 때 poly_init 시각화
    """
    if 'poly_init' not in output:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    poly_init = output['poly_init']
    
    # 색상 팔레트
    colors = [
        (0, 255, 0),    # 초록
        (255, 0, 0),    # 빨강
        (0, 0, 255),    # 파랑
        (255, 255, 0),  # 노랑
        (255, 0, 255),  # 자홍
        (0, 255, 255),  # 청록
        (128, 0, 255),  # 보라
        (255, 128, 0),  # 주황
        (0, 128, 255),  # 하늘
        (128, 255, 0),  # 연두
    ]
    
    # 이미지를 unnormalize (정규화 해제)
    def unnormalize_to_bgr(img_tensor):
        # config에서 mean, std 가져오기
        mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(3, 1, 1)
        
        # 텐서를 numpy로 변환 후 정규화 해제
        img_np = img_tensor.detach().cpu().numpy()  # (C, H, W)
        unnormalized_img = img_np * std + mean  # 정규화 해제
        img_np = unnormalized_img.transpose(1, 2, 0)  # (H, W, C)
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)  # 0-255 범위로 스케일링
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # OpenCV용 BGR 변환
    
    batch_size = batch['inp'].size(0)
    img_names = batch['meta']['img_name'] if 'meta' in batch else ['unknown'] * batch_size
    
    # Batch index로 이미지 분리
    if 'batch_ind' in output:
        batch_indices = output['batch_ind'].cpu().numpy()
    else:
        # batch_ind가 없으면 poly_init의 개수를 batch_size로 나누어 추정
        num_polys = poly_init.size(0)
        batch_indices = np.repeat(np.arange(batch_size), num_polys // batch_size)
    
    down_ratio = 4  # 기본값
    
    for b_idx in range(batch_size):
        # 모든 batch index 처리 (batch_idx 제한 제거)
            
        # 해당 배치의 poly_init만 필터링
        mask = batch_indices == b_idx
        if not np.any(mask):
            continue
            
        b_poly_init = poly_init[mask]
        
        # 원본 이미지 준비
        img_bgr = unnormalize_to_bgr(batch['inp'][b_idx])
        
        # poly_init은 이미 image scale (down_ratio가 적용됨)
        b_poly_init_img = b_poly_init  # (N, V, 2) - 이미 image coordinates
        
        # poly_init 그리기
        for poly_idx, poly in enumerate(b_poly_init_img.cpu().numpy()):
            color = colors[poly_idx % len(colors)]
            
            # 점들을 연결해서 그리기
            poly_int = poly.astype(np.int32)
            cv2.polylines(img_bgr, [poly_int], isClosed=True, color=color, thickness=2)
            
            # 시작점에 원 표시
            cv2.circle(img_bgr, tuple(poly_int[0]), radius=4, color=(255, 255, 255), thickness=-1)
            cv2.circle(img_bgr, tuple(poly_int[0]), radius=3, color=color, thickness=-1)
        
        # 파일명 설정
        img_name = img_names[b_idx] if b_idx < len(img_names) else f"img_{b_idx}"
        save_path = os.path.join(save_dir, f"{img_name}_poly_init.jpg")
        
        # 이미지 저장
        cv2.imwrite(save_path, img_bgr)
        print(f"[Stage 1 Init Viz] Saved: {save_path} (poly_count: {len(b_poly_init_img)})")

def visualize_pixel_ct_alignment(batch, output, save_dir="pixel_ct_viz", batch_idx=0, ct_score_threshold=0.05):
    """
    기존 시각화 방식을 그대로 사용하여 pixel head + ct_hm + polygon을 overlay
    """
    if 'pixel' not in output or 'ct_hm' not in output or 'py' not in output:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    # ✅ 기존 시각화와 동일한 방식으로 polygon 매칭
    if 'detection' not in output or 'batch_ind' not in output:
        return
        
    # ✅ 기존 코드와 동일한 score threshold 적용
    all_detections = output['detection']
    if 'snake' in output.get('task', ''):
        score_idx = 4
    elif 'maskinit' in str(output.get('task', '')):
        score_idx = 0
    else:
        score_idx = 2
    
    # ✅ confidence threshold로 필터링 (vis_ct_score 대신 낮은 값 사용)
    scores = all_detections[:, score_idx]
    vis_mask = scores >= ct_score_threshold  # 낮은 임계값으로 더 많은 polygon 표시
    
    # ✅ 기존 코드와 동일한 align 방식
    py_final = output['py'][-1] if isinstance(output['py'], list) else output['py']
    
    # NMS 결과가 있으면 사용, 없으면 원본 사용
    if 'detection_nms' in output and 'batch_ind_nms' in output:
        if py_final.shape[0] == output['detection_nms'].shape[0]:
            batch_ind = output['batch_ind_nms']
            scores = output['detection_nms'][:, score_idx]
        else:
            batch_ind = output['batch_ind']
    else:
        batch_ind = output['batch_ind']

    vis_mask = scores >= ct_score_threshold
    polys_vis = py_final[vis_mask].detach().cpu()
    bi_vis = batch_ind[vis_mask].detach().cpu()
    
    # ✅ 정규화 해제 (unnormalize)
    def unnormalize_to_bgr(img_tensor):
        # config에서 mean, std 가져오기
        mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(3, 1, 1)
        
        # 텐서를 numpy로 변환 후 정규화 해제
        img_np = img_tensor.detach().cpu().numpy()  # (C, H, W)
        unnormalized_img = img_np * std + mean  # 정규화 해제
        img_np = unnormalized_img.transpose(1, 2, 0)  # (H, W, C)
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)  # 0-255 범위로 스케일링
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # OpenCV용 BGR 변환
    
    batch_size = batch['inp'].size(0)
    img_names = batch['meta']['img_name'] if 'meta' in batch else ['unknown'] * batch_size
    
    # 배치의 모든 이미지에 대해 처리
    for b_idx in range(batch_size):
        try:
            base_img = unnormalize_to_bgr(batch['inp'][b_idx])
            H, W = base_img.shape[:2]
            
            # ✅ 기존 방식과 동일한 polygon 매칭
            idxs = (bi_vis == b_idx).nonzero(as_tuple=True)[0]
            polys_for_img = polys_vis[idxs] if idxs.numel() > 0 else polys_vis[:0]
            
            # Overlay 생성 - 우선순위: ct_hm(투명) > contour > pixel_map
            overlay = base_img.copy()
            
            # 1) Pixel map (초록색) - 가장 아래 레이어
            if 'pixel' in output:
                pixel_i = output['pixel'][b_idx] if not isinstance(output['pixel'], list) else output['pixel'][-1][b_idx]
                pixel_i = pixel_i.detach().cpu()
                if pixel_i.shape[0] > 1:
                    pixel_prob = torch.softmax(pixel_i, dim=0)[1]
                else:
                    pixel_prob = torch.sigmoid(pixel_i[0])
                pixel_resized = torch.nn.functional.interpolate(
                    pixel_prob.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode='nearest'
                ).squeeze()
                pixel_mask = pixel_resized.numpy() > 0.5
                overlay[pixel_mask] = [0, 255, 0]
            
            # 2) Contour 그리기 (밝은 청록색) - 중간 레이어
            for i, poly in enumerate(polys_for_img):
                if poly.numel() == 0:
                    continue
                poly_np = poly.numpy().astype(np.int32)
                cv2.polylines(overlay, [poly_np], isClosed=True, color=(0, 165, 255), thickness=3)  # 주황색(Orange), 두께 증가
            
            # 3) CT heatmap (빨간색, 투명도 적용) - 가장 위 레이어
            if 'ct_hm' in output:
                ct_i = output['ct_hm'][b_idx].detach().cpu()
                if ct_i.shape[0] > 1:
                    ct_prob = ct_i.max(dim=0)[0]
                else:
                    ct_prob = ct_i[0]
                ct_prob_normalized = torch.sigmoid(ct_prob)
                ct_resized = torch.nn.functional.interpolate(
                    ct_prob_normalized.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode='nearest'
                ).squeeze()
                
                ct_np = ct_resized.numpy()
                # 0에 가까우면 완전 투명, 1에 가까우면 불투명하게 처리
                ct_threshold = 0.05  # 최소 임계값 낮춤
                ct_alpha = np.clip(ct_np, 0, 1)  # 0~1 범위로 클리핑
                
                # Vectorized alpha blending for better performance
                # CT heatmap이 임계값 이상인 픽셀에만 투명도 적용
                valid_mask = ct_alpha > ct_threshold
                if np.any(valid_mask):
                    # 3D alpha 텐서 생성 (H, W, 3)
                    alpha_3d = np.stack([ct_alpha] * 3, axis=-1)
                    red_color = np.array([0, 0, 255], dtype=np.uint8)
                    
                    # valid_mask 영역에만 alpha blending 적용
                    # overlay[valid] = (1-alpha) * overlay[valid] + alpha * red
                    overlay[valid_mask] = (
                        (1 - alpha_3d[valid_mask]) * overlay[valid_mask] + 
                        alpha_3d[valid_mask] * red_color
                    ).astype(np.uint8)
            
            # 저장
            img_name = os.path.basename(str(img_names[b_idx]))
            stem = os.path.splitext(img_name)[0]
            save_path = os.path.join(save_dir, f"{stem}_b{batch_idx:04d}_i{b_idx:02d}_overlay.png")
            cv2.imwrite(save_path, overlay)
            
        except Exception as e:
            print(f"[WARN] Visualization failed for image {b_idx}: {e}")
            continue

# 중복된 함수 정의 제거됨 - self_intersection_viz.py에서 import하여 사용

def get_cfg(args):
    cfg = importlib.import_module('configs.' + args.config_file).config
    cfg.test.ct_score = args.ct_score
    cfg.test.with_nms = bool(args.with_nms)
    if cfg.test.with_nms:
        cfg.test.nms_iou_th = args.nms_iou_th
        cfg.test.nms_containment_th = args.nms_containment_th
    cfg.test.segm_or_bbox = args.eval
    cfg.test.test_stage = args.stage
    cfg.test.use_vertex_reduction = not args.not_use_vertex_reduction
    cfg.test.check_simple = args.check_simple
    cfg.test.get_featuremap = args.save_featuremap
    cfg.test.calc_deform_metric = args.with_deform_metric
    cfg.test.th_iou = args.th_iou
    cfg.test.th_score_vertex_cls = float(args.th_score_vertex_cls)
    cfg.test.reduce_apply_adaptive_th = args.reduce_apply_adaptive
    cfg.test.use_rotate_tta = args.rotate_tta
    cfg.test.visualize = args.viz  # ✅ --viz 옵션 추가
    cfg.test.viz_mode = args.viz_mode
    # ✅ 시각화 및 simple ratio 계산을 위한 별도 임계값 설정
    cfg.test.vis_ct_score = args.vis_ct_score
    cfg.test.simple_ratio_cts = args.simple_ratio_cts
    if args.single_rotate_angle is not None:
        cfg.test.single_rotate_angle = args.single_rotate_angle
    if args.dataset != 'None':
        cfg.test.dataset = args.dataset
    if args.exp == 'None':
        if args.checkpoint:  # checkpoint가 있을 때만 경로에서 exp 추출
            sub_dir = args.checkpoint[len(cfg.commen.model_dir)+1:-4]
            args.exp = sub_dir[len(args.config_file)+1:]
        else:
            # checkpoint가 없으면 기본 exp 사용
            args.exp = 't0'  # 또는 다른 기본값
            sub_dir = f'{args.config_file}/{args.exp}'
    else:
        sub_dir = f'{args.config_file}/{args.exp}'

    sub_dir += '/with_nms' if cfg.test.with_nms else '/without_nms'

    cfg.test.print_only_deform_metric = args.print_only_deform_metric
    # ✅ 테스트 시에도 num_workers를 CLI 인자로 설정할 수 있도록 추가합니다.
    if args.num_workers is not None:
        cfg.test.num_workers = args.num_workers
    # ✅ 테스트 배치 사이즈를 CLI 인자로 받아 GPU 수만큼 나눠서 설정합니다.
    if args.test_bs is not None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        cfg.test.batch_size = int(int(args.test_bs) / num_gpus)

    #edit:feat:self-intersection-count:25-08-10
    cfg.test.track_self_intersection = args.track_self_intersection
    cfg.test.viz_pixel_ct = args.viz_pixel_ct
    # ✅ config 파일의 기존 설정을 CLI 인자로 덮어쓰지 않고 OR 조건으로 처리
    cfg.test.viz_pixel_initial_contours = getattr(cfg.test, 'viz_pixel_initial_contours', False) or args.viz_pixel_initial_contours
    # ✅ Pixel map visualization 설정 추가
    cfg.test.viz_pixel_maps = args.viz_pixel_maps
    cfg.test.viz_pixel_mode = args.viz_pixel_mode
    cfg.test.viz_pixel_raw = args.viz_pixel_raw
    cfg.test.viz_snake_energy = args.viz_snake_energy
    # ✅ Vertex reordering 설정 추가
    cfg.test.with_vertex_reordering = args.with_vertex_reordering
    cfg.test.vertex_reorder_method = args.vertex_reorder_method
    print(f"[DEBUG] with_vertex_reordering: {args.with_vertex_reordering}, method: {args.vertex_reorder_method}")
    
    # ✅ Vertex reordering 설정을 경로에 반영 (설정 완료 후)
    if cfg.test.with_vertex_reordering:
        sub_dir += f'/vertex_reorder_{cfg.test.vertex_reorder_method}'
    else:
        sub_dir += '/no_vertex_reorder'
    
    # ✅ Stage 1 poly_init visualization 설정 추가
    cfg.test.viz_stage1_init = args.viz_stage1_init
    # ✅ Self-intersection visualization 설정 추가
    cfg.test.viz_self_intersection = args.viz_self_intersection
    cfg.test.viz_si_save_all = args.viz_si_save_all
    # ✅ Self-intersection CSV logging 옵션 추가
    cfg.test.log_si_to_csv = args.log_si_to_csv
    # ✅ Contour 시각화 두께 설정
    cfg.test.viz_line_thickness = args.viz_line_thickness

    cfg.commen.result_dir = f'{cfg.commen.result_dir}/{sub_dir}'
    cfg.commen.record_dir = f'{cfg.commen.record_dir}/{sub_dir}'
    cfg.commen.model_dir = f'{cfg.commen.model_dir}/{sub_dir}'
    cfg.commen.seed = args.seed
    cfg.commen.deterministic_mode = args.deterministic
    return cfg

class LightningTester(pl.LightningModule):
    def __init__(self, cfg, checkpoint_path):
        super().__init__()
        self.cfg = cfg
        self.network = make_network.get_network(cfg)
        load_network_lightning(self.network, checkpoint_path)
        self.network.eval()
        
        # ✅ Temperature 저장을 위한 checkpoint 경로 저장
        self.checkpoint_path = checkpoint_path
        
        # ✅ Stage 1 감지: config나 checkpoint 경로에서 stage 정보 확인
        self.is_stage1 = False
        if hasattr(cfg.train, 'stage') and int(cfg.train.stage) == 1:
            self.is_stage1 = True
        elif 's1' in checkpoint_path.lower() or 'stage1' in checkpoint_path.lower():
            self.is_stage1 = True
            
            
        if self.is_stage1:
            print("[TEST] Stage 1 모드: pixel IoU만 계산")
            self.evaluator = None
            self.evaluator_b = None
            self.evaluator_pix = SegmentationMetrics()
            # Stage 1에서는 pixel IoU 누적용 변수
            self._pix_inter = 0
            self._pix_union = 0
        else:
            print("[TEST] Stage 2 모드: COCO evaluation 사용")
            self.evaluator = make_evaluator(cfg)
            self.evaluator_b = make_evaluator(cfg, format='bound')
            self.evaluator_pix = SegmentationMetrics()
            
        # ✅ 여러 임계값에 대한 simple ratio 통계를 저장할 딕셔너리를 초기화합니다.
        if not self.is_stage1 and self.cfg.test.check_simple and hasattr(self.cfg.test, 'simple_ratio_cts'):
            self.simple_ratio_stats = {ct: {'simple': 0, 'total': 0} for ct in self.cfg.test.simple_ratio_cts}
        # ✅ 메모리 누수 방지: 전체 output 저장 대신 플래그만 저장
        self.has_detection = False
        self.has_pixel = False

        # === edit:feat:self-intersection-count:25-08-10 ===
        # self-intersection 3단계 카운트
        self.si3_counts = {'at_init': 0, 'at_coarse': 0, 'at_final': 0, 'always_simple': 0, 'total': 0}
        # (원하면 전체 리스트도 모으기)
        self.si3_all = {'init': [], 'coarse': [], 'final': [], 'file_name': []}
        
        # ✅ Self-intersection CSV logging용 리스트
        self.si_log_entries = []  # [(img_name, contour_idx, x_min, y_min, x_max, y_max, num_intersections), ...]

    def on_test_epoch_start(self):
        # ✅ DDP 환경에서 테스트 데이터 로더의 재현성을 보장합니다.
        for loader in self.trainer.test_dataloaders:
            if isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
                loader.sampler.set_epoch(self.current_epoch)

    def test_step(self, batch, batch_idx):
        # print(f"[DEBUG] rank={torch.distributed.get_rank()} batch keys={list(batch.keys())}")
        # ✅ meta가 없거나 None이면 기본 dict 넣기
        if ('meta' not in batch) or (batch['meta'] is None):
            bsz = next(v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor))
            batch['meta'] = {
                'mode': ['test'] * bsz,
                'ct_num': torch.zeros(bsz, dtype=torch.int64, device=batch['inp'].device if 'inp' in batch else 'cpu'),
                'img_name': ['unknown'] * bsz
            }

        # visualize GT before inference
        # visualize_gt(batch, save_dir=f"{self.cfg.commen.result_dir}/debug_gt")

        inp = batch['inp'].cuda()
        with torch.no_grad():
            if self.cfg.test.use_rotate_tta:
                output = self.network.inference_with_rotation_augmentation(inp, batch)
                output['py'].append(output['py_rotate_tta'])
            else:
                output = self.network(inp, batch=batch)
                
                
        # ✅ Stage 1 전용 처리: pixel IoU만 계산하고 early return
        if self.is_stage1:
            if 'pixel' in output:
                # pixel map: (B, C/H, H, W)
                pix = output['pixel'][-1] if isinstance(output['pixel'], list) else output['pixel']
                if pix.shape[1] == 1:
                    prob = torch.sigmoid(pix)  # (B,1,H,W)
                    th = float(getattr(self.cfg.test, 'pixel_th', 0.5))
                    pred = (prob >= th).to(torch.bool)
                else:
                    pred = (torch.argmax(pix, dim=1, keepdim=True) != 0)  # (B,1,H,W) fg!=bg

                gt = (batch['pixel_gt'].unsqueeze(1).to(pred.device) > 0)
                # 해상도 맞추기
                if gt.shape[-2:] != pred.shape[-2:]:
                    gt = torch.nn.functional.interpolate(gt.float(), size=pred.shape[-2:], mode='nearest').to(torch.bool)

                inter = (pred & gt).sum().item()
                union = (pred | gt).sum().item()
                self._pix_inter += inter
                self._pix_union += union
                
                self.has_pixel = True
            
            # ✅ Stage 1 시각화 기능 추가
            if getattr(self.cfg.test, 'visualize', False) or getattr(self.cfg.test, 'viz_pixel_ct', False):
                try:
                    viz_dir = f"{self.cfg.commen.result_dir}/stage1_viz"
                    visualize_stage1_pixel_results(batch, output, save_dir=viz_dir, batch_idx=batch_idx)
                except Exception as e:
                    print(f"[WARN] Stage 1 visualization failed: {e}")
            
            # ✅ Stage 1에서도 pixel initial contour 시각화 체크
            if getattr(self.cfg.test, 'viz_pixel_initial_contours', False) and 'pixel_initial_contours' in output:
                try:
                    visualize_pixel_initial_contours(batch, output, save_dir=f"{self.cfg.commen.result_dir}/pixel_initial_contours_viz", batch_idx=batch_idx)
                except Exception as e:
                    print(f"[WARN STAGE1] Pixel initial contour visualization skipped due to error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ✅ Stage 1에서 poly_init 시각화 (ct_hm 또는 wh head 학습 시)
            if getattr(self.cfg.test, 'viz_stage1_init', False) and 'poly_init' in output:
                # Stage 1에서 ct_hm 또는 wh head가 학습되었는지 확인
                stage1_train_ct_hm = getattr(self.cfg.train, 'stage1_train_ct_hm', False)
                stage1_train_wh = getattr(self.cfg.train, 'stage1_train_wh', False)
                
                if stage1_train_ct_hm or stage1_train_wh:
                    try:
                        visualize_stage1_poly_init(batch, output, save_dir=f"{self.cfg.commen.result_dir}/init")
                    except Exception as e:
                        print(f"[WARN STAGE1] Poly_init visualization skipped due to error: {e}")
                        import traceback
                        traceback.print_exc()
            
            # ✅ Stage 1에서도 Pixel Map 시각화 (e2ec 태스크 등에서 pixel map 있을 때)
            if 'pixel' in output:
                viz_pixel_maps = getattr(self.cfg.test, 'viz_pixel_maps', False)
                viz_pixel_raw = getattr(self.cfg.test, 'viz_pixel_raw', False)
                
                # 일반 pixel map 시각화 (raw 옵션이 없을 때만)
                if viz_pixel_maps and not viz_pixel_raw:
                    try:
                        viz_mode = getattr(self.cfg.test, 'viz_pixel_mode', 'final')
                        save_dir = f"{self.cfg.commen.result_dir}/pixel_maps_{viz_mode}_stage1"
                        visualize_pixel_maps(batch, output, save_dir=save_dir, batch_idx=batch_idx, mode=viz_mode, cfg=self.cfg)
                    except Exception as e:
                        print(f"[WARN STAGE1] Pixel map visualization skipped due to error: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Raw pixel map 시각화 (별도 옵션)
                if viz_pixel_raw:
                    try:
                        viz_mode = getattr(self.cfg.test, 'viz_pixel_mode', 'final')
                        save_dir = f"{self.cfg.commen.result_dir}/pixel_maps_{viz_mode}_raw_stage1"
                        visualize_pixel_maps(batch, output, save_dir=save_dir, batch_idx=batch_idx, mode=viz_mode, cfg=self.cfg)
                    except Exception as e:
                        print(f"[WARN STAGE1] RAW pixel map visualization skipped due to error: {e}")
                        import traceback
                        traceback.print_exc()
            
            # ✅ Stage 1에서는 메모리 정리하고 early return
            del output, inp
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            return None

        #------- edit:feat:self-intersection-count:25-08-10 ---------
        if self.cfg.test.track_self_intersection:
            # === 3-stage SI 집계 ===
            if 'simple_flags' in output:
                m_final = output['simple_flags']['final']  # (N,)
                N = int(m_final.numel())
                if N == 0:
                    # 최종 인스턴스가 없는 배치 → total 증가 없음 (정상)
                    pass
                else:
                    # 필요 시 기본값(True)로 대체
                    m_init = output['simple_flags'].get('init', None)
                    m_coarse = output['simple_flags'].get('coarse', None)
                    if m_init is None:
                        m_init = torch.ones(N, dtype=torch.bool, device=m_final.device)
                    if m_coarse is None:
                        m_coarse = torch.ones(N, dtype=torch.bool, device=m_final.device)

                    # 길이 안전 가드
                    M = min(N, int(m_init.numel()), int(m_coarse.numel()))
                    if M > 0:
                        m_init, m_coarse, m_final = m_init[:M], m_coarse[:M], m_final[:M]

                        # ✨ file_name 배열 만들기 (인스턴스 순서와 정렬 일치 가정)
                        names_np = None
                        try:
                            # 우선적으로 batch_ind 사용 (각 인스턴스 -> 이미지 인덱스 매핑)
                            if ('batch_ind' in output) and isinstance(output['batch_ind'], torch.Tensor) and \
                                    output['batch_ind'].numel() >= M:
                                bi = output['batch_ind'][:M].detach().cpu().numpy()
                                img_names = batch.get('meta', {}).get('img_name',
                                                                      ['unknown'] * int(bi.max() + 1))
                                # 텐서일 수도 있으니 문자열화
                                if isinstance(img_names, torch.Tensor):
                                    img_names = [str(x) for x in img_names]
                                names_np = np.array([str(img_names[int(b)]) for b in bi], dtype=object)
                        except Exception:
                            names_np = None

                        if names_np is None:
                            # 폴백: 배치가 1장이거나 batch_ind가 없을 때
                            img_names = batch.get('meta', {}).get('img_name', ['unknown'])
                            if isinstance(img_names, torch.Tensor):
                                img_names = [str(x) for x in img_names]
                            default_name = str(img_names[0]) if len(img_names) > 0 else 'unknown'
                            names_np = np.array([default_name] * M, dtype=object)

                        # 카테고리 분류
                        init_bad = ~m_init
                        coarse_bad = ~m_coarse
                        final_bad = ~m_final

                        cat_init = init_bad
                        cat_coarse = (~init_bad) & (coarse_bad)
                        cat_final = (~init_bad) & (~coarse_bad) & (final_bad)
                        cat_simple = (~init_bad) & (~coarse_bad) & (~final_bad)

                        # 누적
                        self.si3_counts['at_init'] += int(cat_init.sum().item())
                        self.si3_counts['at_coarse'] += int(cat_coarse.sum().item())
                        self.si3_counts['at_final'] += int(cat_final.sum().item())
                        self.si3_counts['always_simple'] += int(cat_simple.sum().item())
                        self.si3_counts['total'] += int(M)

                        # (옵션) 전체 리스트 저장
                        self.si3_all['init'].append(m_init.detach().cpu().numpy().astype(np.bool_))
                        self.si3_all['coarse'].append(m_coarse.detach().cpu().numpy().astype(np.bool_))
                        self.si3_all['final'].append(m_final.detach().cpu().numpy().astype(np.bool_))
                        self.si3_all['file_name'].append(names_np)  # ✨ add
                        
                        # ✅ self-intersection 변수들 메모리 정리
                        del m_init, m_coarse, m_final, names_np
                        del init_bad, coarse_bad, final_bad
                        del cat_init, cat_coarse, cat_final, cat_simple
            # ------- edit:feat:self-intersection-count:25-08-10 ---------
            else:
                # 후처리(vertex reduction) 적용 시, 원본 백업 후 최종 예측을 대체
                if self.cfg.model.use_vertex_classifier and ('py_reduced' in output) and (self.cfg.test.use_vertex_reduction):
                    output['py_raw'] = output['py'][-1]  # 원본 최종 예측을 'py_raw'에 백업
                    output['py'][-1] = output['py_reduced']  # 평가를 위해 최종 예측을 후처리 결과로 대체
                    if 'is_simple' in output:
                        output['is_simple'] = output['is_simple_reduced']

        if not self.cfg.test.track_self_intersection:
                if self.cfg.test.get_featuremap:
                    dict_save = {k: v.cpu().numpy() for k,v in output['fm'].items()}
                    dict_save.update({'file_name': batch['meta']['img_name']})
                    os.makedirs(f"{self.cfg.commen.result_dir}/test_fm", exist_ok=True)
                    savemat(f"{self.cfg.commen.result_dir}/test_fm/{batch['meta']['img_name'][0].split('.')[0]}.mat", dict_save)

                # ✅ Vertex Re-ordering 후처리: self-intersection 해결
                if self.cfg.test.with_vertex_reordering and 'py' in output and len(output['py']) > 0:
                    from post_process import apply_vertex_reordering_to_output
                    try:
                        output = apply_vertex_reordering_to_output(
                            output, 
                            method=self.cfg.test.vertex_reorder_method,
                            stage='final'
                        )
                        
                        # ✅ Vertex reordering 후 polygon 형식 정규화 (수정됨)
                        # vertex reordering이 polygon 형태를 변경할 수 있으므로 원래 형식으로 복원
                        def normalize_polygon_format(polys):
                            """Vertex reordering 후 polygon을 표준 형식으로 정규화"""
                            if not isinstance(polys, torch.Tensor):
                                return polys
                            
                            # 3D tensor인 경우: vertex reordering 후 복원된 형태 유지
                            if polys.ndim == 3:
                                return polys  # 3D 형태 그대로 유지 (N, V, 2)
                            
                            # 2D tensor인 경우: 형태가 올바른지 확인 (N, 2) 또는 (N*vertices, 2)
                            if polys.ndim == 2 and polys.shape[1] == 2:
                                return polys
                            
                            # 1D인 경우 2D로 reshape
                            if polys.ndim == 1 and polys.numel() % 2 == 0:
                                polys = polys.view(-1, 2)
                            
                            return polys
                        
                        # 모든 py 단계에 정규화 적용
                        if 'py' in output and isinstance(output['py'], list):
                            for i, py_stage in enumerate(output['py']):
                                if py_stage is not None and isinstance(py_stage, torch.Tensor):
                                    output['py'][i] = normalize_polygon_format(py_stage)
                        
                        # 기타 polygon 키들도 정규화
                        for key in ['poly_init', 'poly_coarse']:
                            if key in output and output[key] is not None:
                                if isinstance(output[key], torch.Tensor):
                                    output[key] = normalize_polygon_format(output[key])
                        
                    except Exception as e:
                        print(f"[WARNING] Vertex reordering failed: {e}")

                # ✅ IoU 기반 NMS(Non-Maximum Suppression) 후처리 과정입니다.
                # 동일 이미지 내에서 예측된 인스턴스(polygon)들이 설정된 IoU 임계값 이상으로 겹칠 경우,
                # confidence score가 더 낮은 인스턴스를 제거합니다.
                if self.cfg.test.with_nms and 'detection' in output and output['detection'].numel() > 0:
                    from shapely.geometry import Polygon

                    def poly_iou(p1_np, p2_np):
                        """두 폴리곤의 IoU를 계산합니다. p1_np, p2_np는 numpy 배열입니다."""
                        # ✅ polygon 형식 검증 및 정규화
                        def validate_polygon(p_np):
                            if isinstance(p_np, torch.Tensor):
                                p_np = p_np.detach().cpu().numpy()
                            p_np = np.asarray(p_np)
                            
                            # 차원 정리
                            while p_np.ndim > 2:
                                if p_np.shape[0] == 1:
                                    p_np = p_np[0]
                                elif p_np.shape[-1] == 2:
                                    p_np = p_np.reshape(-1, 2)
                                    break
                                else:
                                    p_np = p_np[0]
                            
                            # 1D인 경우 2D로 변환
                            if p_np.ndim == 1 and p_np.shape[0] % 2 == 0:
                                p_np = p_np.reshape(-1, 2)
                            
                            # 최종 검증
                            if p_np.ndim != 2 or p_np.shape[1] != 2 or p_np.shape[0] < 3:
                                return None
                            
                            return p_np
                        
                        p1_valid = validate_polygon(p1_np)
                        p2_valid = validate_polygon(p2_np)
                        
                        if p1_valid is None or p2_valid is None:
                            return 0.0
                        
                        try:
                            poly1 = Polygon(p1_valid)
                            poly2 = Polygon(p2_valid)
                        except Exception:
                            return 0.0
                        if not poly1.is_valid: poly1 = poly1.buffer(0)
                        if not poly2.is_valid: poly2 = poly2.buffer(0)
                        if not poly1.is_valid or not poly2.is_valid:
                            return 0.0
                        intersect_area = poly1.intersection(poly2).area
                        union_area = poly1.union(poly2).area
                        iou = intersect_area / union_area if union_area > 0 else 0.0
                        return iou

                    def containment_ratio(p_small, p_large):
                        # ✅ polygon 형식 검증 및 정규화 (poly_iou와 동일)
                        def validate_polygon(p_np):
                            if isinstance(p_np, torch.Tensor):
                                p_np = p_np.detach().cpu().numpy()
                            p_np = np.asarray(p_np)
                            
                            # 차원 정리
                            while p_np.ndim > 2:
                                if p_np.shape[0] == 1:
                                    p_np = p_np[0]
                                elif p_np.shape[-1] == 2:
                                    p_np = p_np.reshape(-1, 2)
                                    break
                                else:
                                    p_np = p_np[0]
                            
                            # 1D인 경우 2D로 변환
                            if p_np.ndim == 1 and p_np.shape[0] % 2 == 0:
                                p_np = p_np.reshape(-1, 2)
                            
                            # 최종 검증
                            if p_np.ndim != 2 or p_np.shape[1] != 2 or p_np.shape[0] < 3:
                                return None
                            
                            return p_np
                        
                        p_small_valid = validate_polygon(p_small)
                        p_large_valid = validate_polygon(p_large)
                        
                        if p_small_valid is None or p_large_valid is None:
                            return 0.0
                        
                        try:
                            poly_small = Polygon(p_small_valid)
                            poly_large = Polygon(p_large_valid)
                        except Exception:
                            return 0.0
                        if not poly_small.is_valid:
                            poly_small = poly_small.buffer(0)
                        if not poly_large.is_valid:
                            poly_large = poly_large.buffer(0)
                        if not poly_small.is_valid or not poly_large.is_valid:
                            return 0.0
                        inter = poly_small.intersection(poly_large).area
                        area_small = poly_small.area
                        return inter / area_small if area_small > 0 else 0.0

                    all_detections = output['detection']
                    all_polys = output['py'][-1].cpu().numpy()
                    batch_indices = output['batch_ind']
                    if 'snake' in self.cfg.commen.task:
                        score_idx = 4
                    elif 'maskinit' in self.cfg.commen.task:
                        score_idx = 0
                    else:
                        score_idx = 2
                    all_scores = all_detections[:, score_idx]

                    final_keep_indices = []

                    for b_idx in range(batch['inp'].size(0)):
                        instance_indices = (batch_indices == b_idx).nonzero(as_tuple=True)[0]
                        if instance_indices.numel() < 1:
                            continue

                        # Check bounds before indexing
                        valid_indices = instance_indices[instance_indices < len(all_polys)]
                        if valid_indices.numel() < 1:
                            continue
                        
                        scores_in_img = all_scores[valid_indices]
                        polys_in_img = all_polys[valid_indices.cpu().numpy()]
                        order = scores_in_img.argsort(descending=True)

                        keep_global_indices = []

                        while order.numel() > 0:
                            cur = order[0].item()
                            keep_global_indices.append(valid_indices[cur])  # ✅ Global index

                            if order.numel() == 1:
                                break

                            remaining = order[1:]

                            suppressed_mask = []
                            for r in remaining:
                                poly1 = polys_in_img[cur]
                                poly2 = polys_in_img[r.item()]
                                iou = poly_iou(poly1, poly2)
                                cr1 = containment_ratio(poly1, poly2)
                                cr2 = containment_ratio(poly2, poly1)

                                # ✅ IoU 또는 포함율이 일정 이상이면 중복으로 판단
                                is_duplicate = (
                                        iou > self.cfg.test.nms_iou_th or
                                        cr1 > self.cfg.test.nms_containment_th or cr2 > self.cfg.test.nms_containment_th
                                )
                                suppressed_mask.append(not is_duplicate)

                            suppressed_mask = torch.tensor(suppressed_mask, dtype=torch.bool, device=order.device)
                            order = remaining[suppressed_mask]

                        final_keep_indices.append(torch.stack(keep_global_indices))

                        # if 'add_043_-1r_0' in batch['meta']['img_name'][b_idx]:
                        #     print(f"[NMS DEBUG] img {b_idx}: {len(instance_indices)} → {len(keep_global_indices)} kept")

                    # ✅ 최종 마스크 적용
                    # final_keep_mask = torch.cat(final_keep_indices) if len(final_keep_indices) > 0 else torch.tensor([],
                    #                                                                                                  dtype=torch.long,
                    #                                                                                                  device=self.device)
                    # mask_to_apply = torch.zeros(all_scores.shape[0], dtype=torch.bool, device=self.device)
                    # mask_to_apply[final_keep_mask] = True

                    # ✅ 최종 keep 인덱스들(글로벌)을 score 우선 순서대로 이어붙임
                    if len(final_keep_indices) > 0:
                        keep_idx_sorted = torch.cat(final_keep_indices)  # (K,) 글로벌 인덱스, 점수 내림차순 선택 순서
                    else:
                        keep_idx_sorted = torch.empty(0, dtype=torch.long, device=all_scores.device)

                    # ✅ 원본 유지 + NMS 결과만 append
                    nms_py = output['py'][-1][keep_idx_sorted] if keep_idx_sorted.numel() > 0 else output['py'][-1][:0]
                    output['py'].append(nms_py)

                    # (옵션) 추적/디버깅 위해 선택 인덱스도 보관
                    output['nms_keep_idx'] = keep_idx_sorted

                    # ✅ 나머지 동기화된 뷰를 별도 키로 제공
                    output['detection_nms'] = output['detection'][keep_idx_sorted] if keep_idx_sorted.numel() > 0 \
                        else output['detection'][:0]
                    output['batch_ind_nms'] = output['batch_ind'][keep_idx_sorted] if keep_idx_sorted.numel() > 0 \
                        else output['batch_ind'][:0]
                    output['is_simple_nms'] = output['is_simple'][keep_idx_sorted] if 'is_simple' in output and keep_idx_sorted.numel() > 0 \
                        else output['is_simple'][:0] if 'is_simple' in output else torch.tensor([], dtype=torch.bool)

                    # 🔒 안전장치
                    assert (output['py'][-1].shape[0] == output['detection_nms'].shape[0] ==
                            output['batch_ind_nms'].shape[0]), "NMS 길이 불일치"

                # ✅ Simple Ratio 계산 로직: 여러 임계값에 대해 통계를 누적합니다.
                if self.cfg.test.check_simple and 'is_simple' in output and 'detection' in output and hasattr(
                        self, 'simple_ratio_stats'):
                    # NMS 있으면 NMS 버전 우선 사용
                    det_key = 'detection_nms' if 'detection_nms' in output else 'detection'
                    is_simple_key = 'is_simple_nms' if 'is_simple_nms' in output else 'is_simple'

                    if (det_key in output) and (is_simple_key in output):
                        all_detections = output[det_key]
                        all_is_simple = output[is_simple_key]

                        if 'snake' in self.cfg.commen.task:
                            score_idx = 4
                        elif 'maskinit' in self.cfg.commen.task:
                            score_idx = 0
                        else:
                            score_idx = 2
                        # CPU numpy로 변환
                        scores = all_detections[:, score_idx].detach().cpu().numpy()
                        is_simple_np = all_is_simple.detach().cpu().numpy()

                        # (안전) 길이 동일 확인
                        assert scores.shape[0] == is_simple_np.shape[0], \
                            f"len mismatch in simple stats: scores={scores.shape[0]} simple={is_simple_np.shape[0]}"

                        for ct in self.cfg.test.simple_ratio_cts:
                            mask = (scores >= ct)
                            if mask.any():
                                self.simple_ratio_stats[ct]['simple'] += is_simple_np[mask].sum()
                                self.simple_ratio_stats[ct]['total'] += int(mask.sum())

                if 'detection' in output:
                    # evaluator.evaluate() 들어가기 전
                    output_for_eval = {
                        'py': output['py'],  # evaluator는 리스트를 기대하고 내부에서 [-1] 접근
                        'detection': output.get('detection_nms', output['detection']),
                        'batch_ind': output.get('batch_ind_nms', output['batch_ind']),
                        # 필요시 다른 키들도 동일 규칙으로
                    }

                    self.evaluator.evaluate(output_for_eval, batch)
                    self.evaluator_b.evaluate(output_for_eval, batch)
                else:
                    # pixel_gt를 0-1 범위로 정규화
                    pixel_gt_raw = batch['pixel_gt'].unsqueeze(1).float()
                    pixel_gt_raw = torch.clamp(pixel_gt_raw, min=0, max=1)  # 안전을 위해 클리핑
                    pixel_gt = torch.nn.functional.interpolate(pixel_gt_raw,
                                                               size=output['pixel'].shape[-2:], mode='nearest').squeeze(1)
                    # pixel_gt을 0 또는 1로 정규화 (> 0.5 이면 1, 아니면 0)
                    pixel_gt = (pixel_gt > 0.5).long()
                    self.evaluator_pix.stack_results(pixel_gt, output['pixel'])

        if not self.cfg.test.track_self_intersection:
            # =====================================================================
            # 시각화
            # =====================================================================
            # ✅ Pixel 태스크: contour 시각화는 건너뛰고 pixel map 시각화만 수행
            is_pixel_task = hasattr(self.cfg.commen, 'task') and self.cfg.commen.task == 'pixel'
            
            if self.cfg.test.visualize and 'py' in output and 'batch_ind' in output and not is_pixel_task:
                # 결과 저장 디렉토리
                viz_dir = f"{self.cfg.commen.result_dir}/viz-{self.cfg.test.vis_ct_score}-{self.cfg.test.viz_mode}"
                os.makedirs(os.path.join(viz_dir), exist_ok=True)

                # 색상 팔레트 (형광색/눈에 잘 띄는 색상만 - 파란색, 회색, 어두운색 제외)
                color_palette = [
                    (0, 255, 0),     # 형광 초록
                    (0, 255, 255),   # 형광 노랑 (BGR: cyan -> 화면에서 노랑)
                    (255, 0, 255),   # 형광 마젠타/핑크
                    (0, 255, 128),   # 형광 연두
                    (128, 255, 0),   # 형광 라임
                    (255, 0, 128),   # 형광 핫핑크
                    (0, 128, 255),   # 형광 오렌지 (BGR)
                    (255, 128, 0),   # 형광 시안/청록
                    (50, 255, 255),  # 밝은 노랑
                    (0, 200, 255),   # 밝은 오렌지
                    (200, 0, 255),   # 밝은 핑크
                    (0, 255, 200),   # 밝은 연두
                    (255, 50, 255),  # 밝은 마젠타
                    (100, 255, 100), # 연한 형광 초록
                    (100, 255, 255), # 연한 형광 노랑
                ]

                # 시각화에 적용할 score threshold 마스크
                all_detections = output['detection']
                if 'snake' in self.cfg.commen.task:
                    score_idx = 4
                elif 'maskinit' in self.cfg.commen.task:
                    score_idx = 0
                else:
                    score_idx = 2
                scores = all_detections[:, score_idx]
                vis_mask = scores >= self.cfg.test.vis_ct_score

                batch_indices = output['batch_ind'][vis_mask].cpu()
                batch_size = batch['inp'].size(0)

                # 스테이지 목록 구성: poly_init → poly_coarse → py[0]..py[-1]
                stages = []
                if self.cfg.test.viz_mode == "timeline":
                    if 'poly_init' in output and output['poly_init'] is not None:
                        stages.append(("init", output['poly_init']))
                    if 'poly_coarse' in output and output['poly_coarse'] is not None:
                        stages.append(("coarse", output['poly_coarse']))
                    for i, py_i in enumerate(output['py']):
                        stages.append((f"py{i}", py_i))
                else:
                    # final 모드: 마지막 py만
                    stages.append(("final", output['py'][-1]))

                # ✅ 어떤 detection/batch_ind와 길이가 맞는지 자동 정렬
                def align_det_and_bi(tensor):
                    # NMS 정렬
                    if ('detection_nms' in output) and isinstance(output['detection_nms'], torch.Tensor):
                        if tensor.shape[0] == output['detection_nms'].shape[0]:
                            return output['detection_nms'], output['batch_ind_nms']
                    # 원본 정렬
                    if ('detection' in output) and isinstance(output['detection'], torch.Tensor):
                        if tensor.shape[0] == output['detection'].shape[0]:
                            return output['detection'], output['batch_ind']
                    # 못 맞추면 None
                    return None, None

                # ✅ (stage_name, polys_vis_cpu, batch_inds_vis_cpu)로 준비
                stage_items = []
                for name, tensor in stages:
                    if (not isinstance(tensor, torch.Tensor)) or tensor.numel() == 0:
                        continue

                    det, bi = align_det_and_bi(tensor)
                    if (det is None) or (bi is None):
                        # 길이 안 맞는 스테이지는 시각화에서 스킵 (init/coarse가 다른 정렬일 수 있음)
                        continue

                    scores = det[:, score_idx]
                    vis_mask = scores >= self.cfg.test.vis_ct_score

                    # 비어있을 수 있으니 안전하게 슬라이싱
                    polys_vis = tensor[vis_mask].detach().cpu()
                    bi_vis = bi[vis_mask].detach().cpu()

                    stage_items.append((name, polys_vis, bi_vis))

                # 유틸: 이미지를 unnormalize
                def unnormalize_to_bgr(img_tensor):
                    mean = torch.tensor(self.cfg.data.mean, device=img_tensor.device).view(3, 1, 1)
                    std = torch.tensor(self.cfg.data.std, device=img_tensor.device).view(3, 1, 1)
                    unnormalized_img = img_tensor * std + mean
                    img_np = unnormalized_img.detach().cpu().numpy().transpose(1, 2, 0)
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                    
                    # ✅ WHU dataset 감지 방법 개선
                    is_whu_dataset = False
                    if hasattr(self.cfg, 'data'):
                        if hasattr(self.cfg.data, 'data_dir') and 'whu' in self.cfg.data.data_dir.lower():
                            is_whu_dataset = True
                        elif hasattr(self.cfg.data, 'dataset') and 'whu' in self.cfg.data.dataset.lower():
                            is_whu_dataset = True
                    
                    if is_whu_dataset:
                        # ✅ WHU dataset: 원본 TIFF와 동일한 RGB 순서 유지
                        # 현재 이미지가 BGR로 반정규화되었으므로 RGB로 변환하여 원본과 일치
                        return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                    else:
                        # 기타 dataset: RGB → BGR 변환
                        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # 유틸: 단일 스테이지 드로잉
                def draw_stage(base_img_bgr, polys_for_img, stage_title=None):
                    canvas = base_img_bgr.copy()
                    for i, poly in enumerate(polys_for_img):
                        try:
                            if poly.numel() == 0:
                                continue
                            
                            # Convert to numpy with proper error handling
                            if isinstance(poly, torch.Tensor):
                                poly_np = poly.detach().cpu().numpy()
                            else:
                                poly_np = np.asarray(poly)
                            
                            # Debug: print original shape for troubleshooting
                            original_shape = poly_np.shape
                            
                            # Handle various tensor shapes with more robust logic
                            while poly_np.ndim > 2:
                                # Keep reducing dimensions until we get 2D
                                if poly_np.shape[0] == 1:
                                    poly_np = poly_np[0]  # Remove batch dimension
                                elif poly_np.shape[-1] == 2:
                                    # If last dimension is 2 (coordinates), flatten others
                                    poly_np = poly_np.reshape(-1, 2)
                                    break
                                else:
                                    # Take first element in first dimension
                                    poly_np = poly_np[0]
                            
                            # Handle 1D arrays
                            if poly_np.ndim == 1:
                                if poly_np.shape[0] % 2 == 0 and poly_np.shape[0] > 0:
                                    poly_np = poly_np.reshape(-1, 2)
                                else:
                                    print(f"[WARN] Cannot reshape 1D array of length {poly_np.shape[0]} to (N, 2), orig: {original_shape}")
                                    continue
                            
                            # Handle 0D arrays (scalars)
                            if poly_np.ndim == 0:
                                print(f"[WARN] Scalar polygon detected, orig: {original_shape}, skipping")
                                continue
                            
                            # Final validation
                            if poly_np.ndim != 2 or poly_np.shape[1] != 2:
                                print(f"[WARN] Invalid polygon shape after processing: {poly_np.shape}, skipping")
                                continue
                            if poly_np.shape[0] < 3:
                                print(f"[WARN] Polygon has less than 3 points ({poly_np.shape[0]}), skipping")
                                continue
                            
                            # Check for NaN/Inf values
                            if not np.all(np.isfinite(poly_np)):
                                print(f"[WARN] Polygon contains NaN/Inf values, skipping")
                                continue
                            
                            # ✅ Scale polygon coordinates based on task type
                            # E2EC and CCP models already output coordinates in original image scale
                            # DeepSnake models need down_ratio scaling
                            if hasattr(self.cfg.commen, 'task'):
                                task = self.cfg.commen.task.lower()
                                if 'e2ec' in task or 'ccp' in task:
                                    # E2EC and CCP models already output coordinates in original image scale
                                    pass  # No scaling needed
                                else:
                                    # DeepSnake and other models need down_ratio scaling
                                    down_ratio = getattr(self.cfg.commen, 'down_ratio', 4)
                                    poly_np = poly_np * down_ratio
                            else:
                                # Default: apply down_ratio scaling for backward compatibility
                                down_ratio = getattr(self.cfg.commen, 'down_ratio', 4)
                                poly_np = poly_np * down_ratio
                            
                            # Convert to proper integer coordinates
                            poly_np = np.round(poly_np).astype(np.int32)
                            
                            # Final OpenCV validation - ensure contiguous array
                            poly_np = np.ascontiguousarray(poly_np)
                            
                            # Verify the array format is what OpenCV expects
                            if poly_np.dtype != np.int32 or not poly_np.flags.c_contiguous:
                                poly_np = np.ascontiguousarray(poly_np, dtype=np.int32)
                            
                            # Critical OpenCV validation before calling polylines
                            # Check exact format: must be (N, 2) with N >= 3, contiguous int32
                            if (poly_np.ndim != 2 or poly_np.shape[1] != 2 or 
                                poly_np.shape[0] < 3 or poly_np.dtype != np.int32 or
                                not poly_np.flags.c_contiguous):
                                print(f"[WARN] OpenCV format validation failed for polygon {i}: "
                                      f"shape={poly_np.shape}, dtype={poly_np.dtype}, "
                                      f"contiguous={poly_np.flags.c_contiguous}")
                                continue
                            
                            # Additional safety: test with cv2.checkVector equivalent
                            try:
                                # Create a copy to test with OpenCV functions first
                                test_poly = poly_np.copy()
                                # This should work if format is correct
                                _ = cv2.pointPolygonTest(test_poly.astype(np.float32), (0, 0), False)
                            except Exception as check_error:
                                print(f"[WARN] OpenCV compatibility test failed for polygon {i}: {check_error}")
                                continue
                            
                            color = color_palette[i % len(color_palette)]
                            line_thickness = getattr(self.cfg.test, 'viz_line_thickness', 2)
                            
                            # Final safe call to polylines with additional error handling
                            try:
                                cv2.polylines(canvas, [poly_np], isClosed=True, color=color, thickness=line_thickness)
                                
                                # Add snake energy text if option is enabled
                                if getattr(self.cfg.test, 'viz_snake_energy', False):
                                    try:
                                        # Convert image to grayscale tensor for energy calculation
                                        img_gray = cv2.cvtColor(base_img_bgr, cv2.COLOR_BGR2GRAY)
                                        img_tensor = torch.from_numpy(img_gray).float() / 255.0
                                        poly_tensor = torch.from_numpy(poly_np).float()
                                        
                                        if poly_tensor.shape[0] >= 3:
                                            energy_dict = snake_energy(img_tensor, poly_tensor)
                                            energy_text = format_energy_text(energy_dict)
                                            
                                            # Find centroid for text placement
                                            centroid_x = int(poly_np[:, 0].mean())
                                            centroid_y = int(poly_np[:, 1].mean())
                                            
                                            # Add background rectangle for better readability
                                            text_size = cv2.getTextSize(energy_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                                            cv2.rectangle(canvas, 
                                                        (centroid_x - text_size[0]//2 - 2, centroid_y - text_size[1] - 2),
                                                        (centroid_x + text_size[0]//2 + 2, centroid_y + 2),
                                                        (0, 0, 0), -1)  # black background
                                            
                                            # Add text
                                            cv2.putText(canvas, energy_text, 
                                                      (centroid_x - text_size[0]//2, centroid_y),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # white text
                                    except Exception:
                                        # Skip energy calculation if failed
                                        pass
                                        
                            except cv2.error as cv_error:
                                print(f"[WARN] OpenCV polylines failed for polygon {i}: {cv_error}")
                                # Fallback: try drawing individual lines
                                try:
                                    for j in range(len(poly_np)):
                                        pt1 = tuple(poly_np[j])
                                        pt2 = tuple(poly_np[(j + 1) % len(poly_np)])
                                        cv2.line(canvas, pt1, pt2, color, thickness=line_thickness)
                                except Exception:
                                    print(f"[WARN] Fallback line drawing also failed for polygon {i}")
                                    continue
                            
                        except Exception as e:
                            print(f"[WARN] Failed to draw polygon {i}: {e}")
                            continue
                    
                    if stage_title:
                        cv2.putText(canvas, stage_title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                                    cv2.LINE_AA)
                    return canvas

                # 배치 단위 저장
                for b_idx in range(batch_size):
                    base_img = unnormalize_to_bgr(batch['inp'][b_idx])
                    H, W = base_img.shape[:2]

                    # 이 이미지에 해당하는 인스턴스 인덱스
                    idxs = (batch_indices == b_idx).nonzero(as_tuple=True)[0]

                    if self.cfg.test.viz_mode == "final":
                        if len(stage_items) == 0:
                            out_img = base_img
                        else:
                            name, polys_vis, bi_vis = stage_items[-1]  # 마지막 스테이지(보통 NMS py)
                            idxs = (bi_vis == b_idx).nonzero(as_tuple=True)[0]
                            polys_for_img = polys_vis[idxs] if idxs.numel() > 0 else polys_vis[:0]
                            out_img = draw_stage(base_img, polys_for_img, stage_title=None)
                    else:
                        panels = []
                        sep = np.full((H, 6, 3), 255, dtype=np.uint8)  # 얇은 흰색 벽
                        for name, polys_vis, bi_vis in stage_items:
                            idxs = (bi_vis == b_idx).nonzero(as_tuple=True)[0]
                            polys_for_img = polys_vis[idxs] if idxs.numel() > 0 else polys_vis[:0]
                            panel = draw_stage(base_img, polys_for_img, stage_title=name)
                            panels.append(panel)
                        if len(panels) == 0:
                            out_img = base_img
                        else:
                            merged = panels[0]
                            for p in panels[1:]:
                                merged = np.concatenate([merged, sep, p], axis=1)
                            out_img = merged

                    img_name = os.path.basename(batch['meta']['img_name'][b_idx])
                    stem, ext = os.path.splitext(img_name)
                    
                    # ✅ Self-intersection CSV logging (final stage의 contour만 검사)
                    if getattr(self.cfg.test, 'log_si_to_csv', False) and len(stage_items) > 0:
                        try:
                            from vertex_reordering import has_self_intersection
                            from self_intersection_viz_corrected import detect_self_intersections
                            
                            # 마지막 스테이지 (final)만 검사
                            _, polys_vis_final, bi_vis_final = stage_items[-1]
                            final_idxs = (bi_vis_final == b_idx).nonzero(as_tuple=True)[0]
                            polys_for_check = polys_vis_final[final_idxs] if final_idxs.numel() > 0 else []
                            
                            # SI가 발생한 contour들의 정보 수집
                            si_contours_for_viz = []  # [(contour_idx, poly_np, intersecting_segments), ...]
                            
                            for contour_idx, poly in enumerate(polys_for_check):
                                if poly.numel() == 0:
                                    continue
                                poly_np = poly.detach().cpu().numpy() if isinstance(poly, torch.Tensor) else np.asarray(poly)
                                
                                # 차원 정리
                                while poly_np.ndim > 2:
                                    if poly_np.shape[0] == 1:
                                        poly_np = poly_np[0]
                                    elif poly_np.shape[-1] == 2:
                                        poly_np = poly_np.reshape(-1, 2)
                                        break
                                    else:
                                        poly_np = poly_np[0]
                                
                                if poly_np.ndim == 1 and poly_np.shape[0] % 2 == 0:
                                    poly_np = poly_np.reshape(-1, 2)
                                
                                if poly_np.ndim != 2 or poly_np.shape[1] != 2 or poly_np.shape[0] < 3:
                                    continue
                                
                                # Self-intersection 검사
                                if has_self_intersection(poly_np):
                                    # BBox 계산
                                    x_min, y_min = poly_np.min(axis=0)
                                    x_max, y_max = poly_np.max(axis=0)
                                    
                                    # 교차점 개수 계산
                                    intersecting_segments = detect_self_intersections(poly_np)
                                    num_intersections = len(intersecting_segments)
                                    
                                    # CSV 로깅 엔트리 추가
                                    self.si_log_entries.append({
                                        'img_name': img_name,
                                        'contour_idx': contour_idx,
                                        'x_min': int(x_min),
                                        'y_min': int(y_min),
                                        'x_max': int(x_max),
                                        'y_max': int(y_max),
                                        'num_intersections': num_intersections
                                    })
                                    
                                    # 시각화용 정보 저장
                                    si_contours_for_viz.append((contour_idx, poly_np.copy(), intersecting_segments))
                            
                            # ✅ SI가 발생한 contour들만 시각화하여 저장
                            if len(si_contours_for_viz) > 0:
                                si_viz_dir = os.path.join(self.cfg.commen.result_dir, "si_contours_viz")
                                os.makedirs(si_viz_dir, exist_ok=True)
                                
                                # 원본 이미지 위에 SI contour들만 그리기
                                si_canvas = base_img.copy()
                                
                                si_line_thickness = getattr(self.cfg.test, 'viz_line_thickness', 2)
                                for c_idx, poly_np, intersecting_segments in si_contours_for_viz:
                                    # Contour 그리기 (빨간색)
                                    poly_int = np.round(poly_np).astype(np.int32)
                                    cv2.polylines(si_canvas, [poly_int], isClosed=True, color=(0, 0, 255), thickness=si_line_thickness)
                                    
                                    # 교차점 표시 (노란색 원)
                                    for seg1_idx, seg2_idx, intersection_point in intersecting_segments:
                                        ix, iy = int(intersection_point[0]), int(intersection_point[1])
                                        cv2.circle(si_canvas, (ix, iy), 5, (0, 255, 255), -1)
                                    
                                    # BBox 그리기 (초록색)
                                    bx_min, by_min = poly_np.min(axis=0).astype(int)
                                    bx_max, by_max = poly_np.max(axis=0).astype(int)
                                    cv2.rectangle(si_canvas, (bx_min, by_min), (bx_max, by_max), (0, 255, 0), 1)
                                    
                                    # Contour index 표시
                                    cv2.putText(si_canvas, f"#{c_idx}", (bx_min, by_min - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                                # SI 개수 표시
                                cv2.putText(si_canvas, f"SI contours: {len(si_contours_for_viz)}", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                
                                # 저장
                                si_save_path = os.path.join(si_viz_dir, f"{stem}_si.png")
                                cv2.imwrite(si_save_path, si_canvas)
                                
                        except Exception as e:
                            print(f"[WARN] Self-intersection CSV logging failed: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # ✅ WHU dataset 감지 방법 개선: dataset 이름으로 확인
                    is_whu_dataset = False
                    if hasattr(self.cfg, 'data'):
                        if hasattr(self.cfg.data, 'data_dir') and 'whu' in self.cfg.data.data_dir.lower():
                            is_whu_dataset = True
                        elif hasattr(self.cfg.data, 'dataset') and 'whu' in self.cfg.data.dataset.lower():
                            is_whu_dataset = True
                    # Dataset 이름으로도 확인
                    if hasattr(batch, 'meta') and 'dataset' in batch['meta'] and 'whu' in str(batch['meta']['dataset']).lower():
                        is_whu_dataset = True
                    # 파일 확장자로도 확인
                    if ext.lower() in ['.tiff', '.tif']:
                        is_whu_dataset = True
                    
                    if is_whu_dataset:
                        # WHU dataset: 무조건 PNG 저장 (RGB 색상 유지)
                        save_name = f"{stem}.png"
                        # RGB 형식으로 저장하기 위해 BGR → RGB 변환
                        out_img_for_save = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(os.path.join(viz_dir, save_name), out_img_for_save)
                    else:
                        save_name = f"{stem}{ext if self.cfg.test.viz_mode == 'final' else '.png'}"
                        cv2.imwrite(os.path.join(viz_dir, save_name), out_img)

            # =====================================================================
            # Pixel Head vs CT Heatmap 공간적 정렬 시각화
            # =====================================================================
            if getattr(self.cfg.test, 'viz_pixel_ct', False) and 'pixel' in output and 'ct_hm' in output:
                try:
                    visualize_pixel_ct_alignment(batch, output, save_dir=f"{self.cfg.commen.result_dir}/pixel_ct_viz", batch_idx=batch_idx, ct_score_threshold=self.cfg.test.ct_score)
                except Exception as e:
                    print(f"[WARN] Pixel-CT visualization skipped due to error: {e}")

            # =====================================================================  
            # Pixel Map Initial Contour 시각화 (ct_score 필터링 전)
            # =====================================================================
            if getattr(self.cfg.test, 'viz_pixel_initial_contours', False) and 'pixel_initial_contours' in output:
                try:
                    visualize_pixel_initial_contours(batch, output, save_dir=f"{self.cfg.commen.result_dir}/pixel_initial_contours_viz", batch_idx=batch_idx)
                except Exception as e:
                    print(f"[WARN] Pixel initial contour visualization skipped due to error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # =====================================================================
            # Pixel Map 시각화 (final 또는 all stages)
            # =====================================================================
            if 'pixel' in output:
                viz_pixel_maps = getattr(self.cfg.test, 'viz_pixel_maps', False)
                viz_pixel_raw = getattr(self.cfg.test, 'viz_pixel_raw', False)
                
                # 일반 pixel map 시각화 (raw 옵션이 없을 때만)
                if viz_pixel_maps and not viz_pixel_raw:
                    try:
                        viz_mode = getattr(self.cfg.test, 'viz_pixel_mode', 'final')
                        save_dir = f"{self.cfg.commen.result_dir}/pixel_maps_{viz_mode}"
                        visualize_pixel_maps(batch, output, save_dir=save_dir, batch_idx=batch_idx, mode=viz_mode, cfg=self.cfg)
                    except Exception as e:
                        print(f"[WARN] Pixel map visualization skipped due to error: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Raw pixel map 시각화 (별도 옵션)
                if viz_pixel_raw:
                    try:
                        viz_mode = getattr(self.cfg.test, 'viz_pixel_mode', 'final')
                        save_dir = f"{self.cfg.commen.result_dir}/pixel_maps_{viz_mode}_raw"
                        visualize_pixel_maps(batch, output, save_dir=save_dir, batch_idx=batch_idx, mode=viz_mode, cfg=self.cfg)
                    except Exception as e:
                        print(f"[WARN] RAW pixel map visualization skipped due to error: {e}")
                        import traceback
                        traceback.print_exc()

            # Self-intersection 시각화
            # =====================================================================
            if getattr(self.cfg.test, 'viz_self_intersection', False) and 'py' in output:
                print(f"[DEBUG] Calling visualize_self_intersections...")
                try:
                    save_dir = f"{self.cfg.commen.result_dir}/self_intersection_viz"
                    print(f"[DEBUG] Save directory for self-intersection viz: {save_dir}")
                    # viz_self_intersection이 켜져있으면 자동으로 모든 SI 케이스 저장
                    # save_individual=True로 각 SI contour를 개별 이미지로도 저장
                    if use_corrected_viz:
                        # 새로운 corrected version은 cfg를 필요로 함
                        visualize_self_intersections_corrected(
                            batch, output, self.cfg,
                            save_dir=save_dir, 
                            batch_idx=batch_idx, 
                            save_individual=True
                        )
                    else:
                        # 이전 version들은 down_ratio 파라미터 사용
                        down_ratio = getattr(self.cfg.commen, 'down_ratio', 4)
                        visualize_self_intersections_improved(
                            batch, output, 
                            save_dir=save_dir, 
                            batch_idx=batch_idx, 
                            save_individual=True,
                            down_ratio=down_ratio
                        )
                except Exception as e:
                    print(f"[WARN] Self-intersection visualization skipped due to error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                pass

        # ✅ 메모리 누수 방지: 전체 output 저장 대신 플래그만 설정
        # Stage 1에서는 detection 플래그 설정하지 않음
        if not self.is_stage1 and 'detection' in output:
            self.has_detection = True
        if 'pixel' in output:
            self.has_pixel = True
        
        # ✅ 메모리 누수 방지: 각 배치 처리 후 GPU 메모리 정리
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # ✅ 중간 변수들 명시적 삭제
        del output
        if 'inp' in locals():
            del inp
    
    def save_temperature_values(self):
        """trainable_softmax 또는 trainable_softmax_softclamp가 설정된 경우 학습된 temperature 값을 저장"""
        print(f"[DEBUG] save_temperature_values called")
        print(f"[DEBUG] hasattr(self.cfg.model, 'ccp_deform_pixel_norm'): {hasattr(self.cfg.model, 'ccp_deform_pixel_norm')}")
        if hasattr(self.cfg.model, 'ccp_deform_pixel_norm'):
            print(f"[DEBUG] self.cfg.model.ccp_deform_pixel_norm: {self.cfg.model.ccp_deform_pixel_norm}")
        
        if not hasattr(self.cfg.model, 'ccp_deform_pixel_norm') or \
           self.cfg.model.ccp_deform_pixel_norm not in ['trainable_softmax', 'trainable_softmax_softclamp']:
            print(f"[DEBUG] Temperature saving skipped - not trainable_softmax type")
            return
            
        print("[INFO] Saving learned temperature values...")
        
        # Evolution 모듈에서 temperature 값 추출
        temperature_values = {}
        
        try:
            import torch.nn.functional as F
            
            print(f"[DEBUG] Network type: {type(self.network)}")
            print(f"[DEBUG] Network has 'evolve': {hasattr(self.network, 'evolve')}")
            print(f"[DEBUG] Network has 'gcn': {hasattr(self.network, 'gcn')}")
            
            # CCP 네트워크에서 evolution 모듈 찾기
            # CCPnet에서는 evolution 모듈이 self.gcn으로 저장됨
            evolve_modules = None
            if hasattr(self.network, 'evolve'):
                evolve_modules = self.network.evolve
                print(f"[DEBUG] Using self.network.evolve")
            elif hasattr(self.network, 'gcn'):
                evolve_modules = self.network.gcn
                print(f"[DEBUG] Using self.network.gcn")
            
            if evolve_modules is not None:
                print(f"[DEBUG] evolve_modules type: {type(evolve_modules)}")
                print(f"[DEBUG] evolve_modules is list: {isinstance(evolve_modules, list)}")
                
                if isinstance(evolve_modules, list):
                    print(f"[DEBUG] evolve_modules list length: {len(evolve_modules)}")
                    for i, evolve_module in enumerate(evolve_modules):
                        print(f"[DEBUG] evolve_module[{i}] type: {type(evolve_module)}")
                        print(f"[DEBUG] evolve_module[{i}] has 'temperature': {hasattr(evolve_module, 'temperature')}")
                        print(f"[DEBUG] evolve_module[{i}] has 'u': {hasattr(evolve_module, 'u')}")
                        
                        # trainable_softmax 타입
                        if hasattr(evolve_module, 'temperature'):
                            temp_tensor = evolve_module.temperature.detach().cpu()
                            # F.softplus(temp) + 1e-8로 실제 사용되는 값 계산
                            activated_tensor = F.softplus(temp_tensor) + 1e-8
                            
                            temp_values = temp_tensor.numpy().tolist()
                            activated_values = activated_tensor.numpy().tolist()
                            
                            temperature_values[f'evolve_iter_{i}'] = {
                                'type': 'trainable_softmax',
                                'raw_values': temp_values,
                                'activated_values': activated_values,
                                'raw_stats': {
                                    'mean': float(temp_tensor.mean().item()),
                                    'std': float(temp_tensor.std().item()),
                                    'min': float(temp_tensor.min().item()),
                                    'max': float(temp_tensor.max().item())
                                },
                                'activated_stats': {
                                    'mean': float(activated_tensor.mean().item()),
                                    'std': float(activated_tensor.std().item()),
                                    'min': float(activated_tensor.min().item()),
                                    'max': float(activated_tensor.max().item())
                                },
                                'shape': list(temp_tensor.shape)
                            }
                            print(f"  Evolution iter {i}: shape={temp_tensor.shape}")
                            print(f"    Raw: mean={temp_tensor.mean():.4f}, std={temp_tensor.std():.4f}")
                            print(f"    Activated: mean={activated_tensor.mean():.4f}, std={activated_tensor.std():.4f}")
                        
                        # trainable_softmax_softclamp 타입
                        elif hasattr(evolve_module, 'u') and hasattr(evolve_module, 'T_lo') and hasattr(evolve_module, 'T_hi'):
                            u_tensor = evolve_module.u.detach().cpu()
                            T_lo = evolve_module.T_lo
                            T_hi = evolve_module.T_hi
                            # T = T_lo + (T_hi - T_lo) * sigmoid(u)로 실제 사용되는 값 계산
                            activated_tensor = T_lo + (T_hi - T_lo) * torch.sigmoid(u_tensor)
                            
                            u_values = u_tensor.numpy().tolist()
                            activated_values = activated_tensor.numpy().tolist()
                            
                            temperature_values[f'evolve_iter_{i}'] = {
                                'type': 'trainable_softmax_softclamp',
                                'T_lo': T_lo,
                                'T_hi': T_hi,
                                'raw_values': u_values,
                                'activated_values': activated_values,
                                'raw_stats': {
                                    'mean': float(u_tensor.mean().item()),
                                    'std': float(u_tensor.std().item()),
                                    'min': float(u_tensor.min().item()),
                                    'max': float(u_tensor.max().item())
                                },
                                'activated_stats': {
                                    'mean': float(activated_tensor.mean().item()),
                                    'std': float(activated_tensor.std().item()),
                                    'min': float(activated_tensor.min().item()),
                                    'max': float(activated_tensor.max().item())
                                },
                                'shape': list(u_tensor.shape)
                            }
                            print(f"  Evolution iter {i}: shape={u_tensor.shape}, T_range=[{T_lo}, {T_hi}]")
                            print(f"    Raw u: mean={u_tensor.mean():.4f}, std={u_tensor.std():.4f}")
                            print(f"    Activated T: mean={activated_tensor.mean():.4f}, std={activated_tensor.std():.4f}")
                else:
                    # 단일 evolution 모듈
                    print(f"[DEBUG] Single evolve_module type: {type(evolve_modules)}")
                    print(f"[DEBUG] Single evolve_module has 'temperature': {hasattr(evolve_modules, 'temperature')}")
                    print(f"[DEBUG] Single evolve_module has 'u': {hasattr(evolve_modules, 'u')}")
                    print(f"[DEBUG] Single evolve_module has 'T_lo': {hasattr(evolve_modules, 'T_lo')}")
                    print(f"[DEBUG] Single evolve_module has 'T_hi': {hasattr(evolve_modules, 'T_hi')}")
                    
                    # trainable_softmax 타입
                    if hasattr(evolve_modules, 'temperature'):
                        temp_tensor = evolve_modules.temperature.detach().cpu()
                        activated_tensor = F.softplus(temp_tensor) + 1e-8
                        
                        temp_values = temp_tensor.numpy().tolist()
                        activated_values = activated_tensor.numpy().tolist()
                        
                        temperature_values['evolve'] = {
                            'type': 'trainable_softmax',
                            'raw_values': temp_values,
                            'activated_values': activated_values,
                            'raw_stats': {
                                'mean': float(temp_tensor.mean().item()),
                                'std': float(temp_tensor.std().item()),
                                'min': float(temp_tensor.min().item()),
                                'max': float(temp_tensor.max().item())
                            },
                            'activated_stats': {
                                'mean': float(activated_tensor.mean().item()),
                                'std': float(activated_tensor.std().item()),
                                'min': float(activated_tensor.min().item()),
                                'max': float(activated_tensor.max().item())
                            },
                            'shape': list(temp_tensor.shape)
                        }
                        print(f"  Evolution: shape={temp_tensor.shape}")
                        print(f"    Raw: mean={temp_tensor.mean():.4f}, std={temp_tensor.std():.4f}")
                        print(f"    Activated: mean={activated_tensor.mean():.4f}, std={activated_tensor.std():.4f}")
                    
                    # trainable_softmax_softclamp 타입
                    elif hasattr(evolve_modules, 'u') and hasattr(evolve_modules, 'T_lo') and hasattr(evolve_modules, 'T_hi'):
                        u_tensor = evolve_modules.u.detach().cpu()
                        T_lo = evolve_modules.T_lo
                        T_hi = evolve_modules.T_hi
                        # T = T_lo + (T_hi - T_lo) * sigmoid(u)로 실제 사용되는 값 계산
                        activated_tensor = T_lo + (T_hi - T_lo) * torch.sigmoid(u_tensor)
                        
                        u_values = u_tensor.numpy().tolist()
                        activated_values = activated_tensor.numpy().tolist()
                        
                        temperature_values['evolve'] = {
                            'type': 'trainable_softmax_softclamp',
                            'T_lo': T_lo,
                            'T_hi': T_hi,
                            'raw_values': u_values,
                            'activated_values': activated_values,
                            'raw_stats': {
                                'mean': float(u_tensor.mean().item()),
                                'std': float(u_tensor.std().item()),
                                'min': float(u_tensor.min().item()),
                                'max': float(u_tensor.max().item())
                            },
                            'activated_stats': {
                                'mean': float(activated_tensor.mean().item()),
                                'std': float(activated_tensor.std().item()),
                                'min': float(activated_tensor.min().item()),
                                'max': float(activated_tensor.max().item())
                            },
                            'shape': list(u_tensor.shape)
                        }
                        print(f"  Evolution: shape={u_tensor.shape}, T_range=[{T_lo}, {T_hi}]")
                        print(f"    Raw u: mean={u_tensor.mean():.4f}, std={u_tensor.std():.4f}")
                        print(f"    Activated T: mean={activated_tensor.mean():.4f}, std={activated_tensor.std():.4f}")
                        
        except Exception as e:
            print(f"[WARN] Failed to extract temperature values: {e}")
            import traceback
            traceback.print_exc()
            return
            
        if not temperature_values:
            print("[WARN] No temperature values found in the model")
            if evolve_modules is not None:
                # 모든 속성 출력해보기
                attrs = [attr for attr in dir(evolve_modules) if not attr.startswith('_')]
                print(f"[DEBUG] Available attributes: {attrs[:10]}...")  # 처음 10개만
                # temperature 관련 속성 찾기
                temp_attrs = [attr for attr in attrs if 'temp' in attr.lower() or 'u' in attr or 'T_' in attr]
                print(f"[DEBUG] Temperature-related attributes: {temp_attrs}")
            return
            
        # 저장 경로 결정
        result_dir = self.cfg.commen.result_dir
        print(f"[DEBUG] result_dir: {result_dir}")
        
        temp_file = os.path.join(result_dir, "learned_temperature_values.txt")
        print(f"[DEBUG] temp_file path: {temp_file}")
        
        # 파일 저장을 위한 디렉토리 생성 (파일 경로의 디렉토리 부분)
        temp_file_dir = os.path.dirname(temp_file)
        print(f"[DEBUG] Creating directory: {temp_file_dir}")
        os.makedirs(temp_file_dir, exist_ok=True)
        
        # 파일에 저장
        try:
            print(f"[DEBUG] Attempting to write to: {temp_file}")
            print(f"[DEBUG] Directory exists: {os.path.exists(temp_file_dir)}")
            print(f"[DEBUG] Directory writable: {os.access(temp_file_dir, os.W_OK)}")
            
            with open(temp_file, 'w') as f:
                f.write("=== Learned Temperature Values ===\n")
                f.write(f"Model: {self.checkpoint_path}\n")
                f.write(f"Config: ccp_deform_pixel_norm = '{self.cfg.model.ccp_deform_pixel_norm}'\n\n")
            
                for module_name, temp_info in temperature_values.items():
                    f.write(f"[{module_name}]\n")
                    f.write(f"Type: {temp_info['type']}\n")
                    f.write(f"Shape: {temp_info['shape']}\n")
                    
                    if temp_info['type'] == 'trainable_softmax':
                        f.write(f"Note: Activated values = F.softplus(raw_values) + 1e-8\n")
                        f.write(f"Raw Statistics:\n")
                        f.write(f"  Mean: {temp_info['raw_stats']['mean']:.6f}\n")
                        f.write(f"  Std:  {temp_info['raw_stats']['std']:.6f}\n")
                        f.write(f"  Min:  {temp_info['raw_stats']['min']:.6f}\n")
                        f.write(f"  Max:  {temp_info['raw_stats']['max']:.6f}\n")
                        f.write(f"Activated Statistics (used in inference):\n")
                    elif temp_info['type'] == 'trainable_softmax_softclamp':
                        f.write(f"Range: T_lo={temp_info['T_lo']}, T_hi={temp_info['T_hi']}\n")
                        f.write(f"Note: Activated values = T_lo + (T_hi - T_lo) * sigmoid(u)\n")
                        f.write(f"Raw u Statistics:\n")
                        f.write(f"  Mean: {temp_info['raw_stats']['mean']:.6f}\n")
                        f.write(f"  Std:  {temp_info['raw_stats']['std']:.6f}\n")
                        f.write(f"  Min:  {temp_info['raw_stats']['min']:.6f}\n")
                        f.write(f"  Max:  {temp_info['raw_stats']['max']:.6f}\n")
                        f.write(f"Activated T Statistics (used in inference):\n")
                    
                    f.write(f"  Mean: {temp_info['activated_stats']['mean']:.6f}\n")
                    f.write(f"  Std:  {temp_info['activated_stats']['std']:.6f}\n")
                    f.write(f"  Min:  {temp_info['activated_stats']['min']:.6f}\n")
                    f.write(f"  Max:  {temp_info['activated_stats']['max']:.6f}\n")
                    
                    if temp_info['type'] == 'trainable_softmax':
                        f.write(f"Raw values: {temp_info['raw_values']}\n")
                    elif temp_info['type'] == 'trainable_softmax_softclamp':
                        f.write(f"Raw u values: {temp_info['raw_values']}\n")
                    
                    f.write(f"Activated values: {temp_info['activated_values']}\n\n")
                
            print(f"✅ Temperature values saved to: {temp_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save temperature values: {e}")
            import traceback
            traceback.print_exc()

    def on_test_epoch_end(self):
        print(f"[DEBUG] on_test_epoch_end called")
        print(f"[DEBUG] self.trainer.is_global_zero: {self.trainer.is_global_zero}")
        if not self.cfg.test.track_self_intersection:
            # ✅ Simple Ratio 계산 및 파일 저장 로직
            if self.cfg.test.check_simple and hasattr(self, 'simple_ratio_stats'):
                # DDP 환경을 위해 모든 프로세스의 통계를 수집합니다.
                stats_tensor_list = []
                # 순서를 보장하기 위해 정렬된 키를 사용합니다.
                sorted_cts = sorted(self.simple_ratio_stats.keys())
                for ct in sorted_cts:
                    stats_tensor_list.extend([self.simple_ratio_stats[ct]['simple'], self.simple_ratio_stats[ct]['total']])

                stats_tensor = torch.tensor(stats_tensor_list, dtype=torch.float32, device=self.device)

                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(stats_tensor, op=torch.distributed.ReduceOp.SUM)

                # 메인 프로세스(rank 0)에서만 파일 출력 및 저장을 수행합니다.
                if self.trainer.is_global_zero:
                    summary_lines = ["--- Simple Ratio Summary ---"]
                    for i, ct in enumerate(sorted_cts):
                        simple_count = stats_tensor[i*2].item()
                        total_count = stats_tensor[i*2 + 1].item()
                        ratio = (simple_count / total_count) if total_count > 0 else 0
                        line = f"simple ratio (ct > {ct:.2f}): {ratio:.4f} ({int(simple_count)}/{int(total_count)})"
                        summary_lines.append(line)
                        print(line)

                    # 결과를 .txt 파일에 저장
                    summary_path = os.path.join(self.cfg.commen.result_dir, "simple_ratio_summary.txt")
                    with open(summary_path, 'w') as f:
                        f.write("\n".join(summary_lines))
                    print(f"Simple ratio summary saved to {summary_path}")

            # ✅ Self-intersection CSV 저장
            if getattr(self.cfg.test, 'log_si_to_csv', False) and self.trainer.is_global_zero:
                if len(self.si_log_entries) > 0:
                    import csv
                    csv_path = os.path.join(self.cfg.commen.result_dir, "self_intersection_contours.csv")
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    with open(csv_path, 'w', newline='') as f:
                        fieldnames = ['img_name', 'contour_idx', 'x_min', 'y_min', 'x_max', 'y_max', 'num_intersections']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for entry in self.si_log_entries:
                            writer.writerow(entry)
                    print(f"[SI-CSV] Self-intersection contours saved to {csv_path}")
                    print(f"[SI-CSV] Total contours with self-intersection: {len(self.si_log_entries)}")
                else:
                    print(f"[SI-CSV] No self-intersection contours found")

            # print(f"[DEBUG/on_test_epoch_end] rank={torch.distributed.get_rank()}")
            # 🚨 DDP HANG BUG FIX: DDP 환경에서는 모든 프로세스가 동기화 지점에 함께 도달해야 합니다.
            # 일부 프로세스만 `summarize`를 호출하고 나머지는 조기 종료하면 교착 상태(deadlock)가 발생합니다.
            # 따라서 모든 프로세스가 평가를 수행할지 여부를 함께 결정해야 합니다.

            # 1. 모든 프로세스가 test_step을 마칠 때까지 대기합니다. (동기화 지점)
            if _is_multi_process():
                torch.distributed.barrier()

            # 2. 'detection' 출력이 있는지 로컬에서 확인합니다.
            #    결과가 있으면 1, 없으면 0으로 플래그를 설정합니다.
            has_detection_local = 1 if self.has_detection else 0
            has_detection_tensor = torch.tensor(has_detection_local, device=self.device)

            # 3. 모든 프로세스의 플래그를 모아(all_reduce) 공유합니다.
            #    ReduceOp.MAX를 사용하여 단 하나의 프로세스라도 결과가 있으면(1), 모든 프로세스가 1을 갖게 됩니다.
            if _is_multi_process():
                torch.distributed.all_reduce(has_detection_tensor, op=torch.distributed.ReduceOp.MAX)

            # 4. 동기화된 플래그를 기반으로 모든 프로세스가 함께 평가를 수행하거나 건너뜁니다.
            if has_detection_tensor.item() == 1:
                summ_dict, mean_ious_diff = self.evaluator.summarize()
                # 로그는 메인 프로세스(rank 0)에서만 기록합니다.
                if self.trainer.is_global_zero:
                    self.log("test_metric", list(summ_dict.values())[0])
                    # 결과를 .txt 파일에 저장
                    metric_path = os.path.join(self.cfg.commen.result_dir, "test_metric.txt")
                    with open(metric_path, 'w') as f:
                        for k, v in summ_dict.items():
                            f.write(f"{k}: {float(v):.4f}\n")
                    print(f"Metrics saved to {metric_path}")
            elif self.has_pixel:
                if self.is_stage1:
                    # Stage 1: pixel IoU만 계산
                    pixel_iou = float(self._pix_inter) / float(self._pix_union + 1e-6)
                    self.log("test_metric", pixel_iou)
                    if self.trainer.is_global_zero:
                        # Stage 1 전용 metric 파일 저장
                        os.makedirs(self.cfg.commen.result_dir, exist_ok=True)
                        metric_path = os.path.join(self.cfg.commen.result_dir, "test_metric.txt")
                        with open(metric_path, 'w') as f:
                            f.write(f"Pixel IoU: {pixel_iou:.4f}\n")
                        print(f"[Stage 1] Pixel IoU: {pixel_iou:.4f}")
                        print(f"Metrics saved to {metric_path}")
                else:
                    pixel_acc, dice, precision, recall = self.evaluator_pix()
                    self.log("pixel_acc", pixel_acc)
                    self.log("dice", dice)
                    self.log("f1", 2 * (precision * recall) / (precision + recall))
                    if self.trainer.is_global_zero:
                        # metric 파일 저장
                        os.makedirs(self.cfg.commen.result_dir, exist_ok=True)
                        metric_path = os.path.join(self.cfg.commen.result_dir, "test_metric.txt")
                        with open(metric_path, 'w') as f:
                            f.write(f"Pixel Accuracy: {pixel_acc:.4f}\n")
                            f.write(f"Dice: {dice:.4f}\n")
                            f.write(f"Precision: {precision:.4f}\n")
                            f.write(f"Recall: {recall:.4f}\n")
                            f.write(f"F1: {2 * (precision * recall) / (precision + recall):.4f}\n")
                        print(f"\nPixel Accuracy: {pixel_acc:.4f}")
                        print(f"Dice: {dice:.4f}")
                        print(f"Precision: {precision:.4f}")
                        print(f"Recall: {recall:.4f}")
                        print(f"F1: {2 * (precision * recall) / (precision + recall):.4f}")
                        print(f"Metrics saved to {metric_path}")
        else:
            # ----- edit:feat:self-intersection-count:25-08-10 -----
            # === 3-stage SI 요약 ===
            c = self.si3_counts
            if c['total'] > 0:
                print("\n[SI-3stage] summary over dataset")
                print(f"  total            : {c['total']}")
                print(f"  at_init          : {c['at_init']}  ({c['at_init'] / c['total']:.4f})")
                print(f"  at_coarse        : {c['at_coarse']}  ({c['at_coarse'] / c['total']:.4f})")
                print(f"  at_final         : {c['at_final']}  ({c['at_final'] / c['total']:.4f})")
                print(f"  always_simple    : {c['always_simple']}  ({c['always_simple'] / c['total']:.4f})")

                # (옵션) 리스트로도 저장
                try:
                    all_init = np.concatenate(self.si3_all['init']) if len(self.si3_all['init']) > 0 else np.array([])
                    all_coarse = np.concatenate(self.si3_all['coarse']) if len(
                        self.si3_all['coarse']) > 0 else np.array([])
                    all_final = np.concatenate(self.si3_all['final']) if len(self.si3_all['final']) > 0 else np.array(
                        [])
                    all_names = np.concatenate(self.si3_all['file_name']) if len(
                        self.si3_all['file_name']) > 0 else np.array([], dtype=object)  # ✨ add

                    out_dir = self.cfg.commen.result_dir
                    os.makedirs(out_dir, exist_ok=True)
                    # ✨ file_name 포함 저장 (object 배열 → 로드시 allow_pickle=True 필요)
                    if (not torch.distributed.is_initialized()) or self.trainer.is_global_zero:
                        np.savez(os.path.join(out_dir, "simple_flags_3stage.npz"),
                                 init=all_init, coarse=all_coarse, final=all_final, file_name=all_names)
                        print(f"  saved flags → {out_dir}/simple_flags_3stage.npz (with file_name)")
                except Exception as e:
                    print(f"  [warn] saving flags failed: {e}")

        # ✅ Temperature 값 저장 (trainable_softmax인 경우)
        print(f"[DEBUG] Before temperature save check - is_global_zero: {self.trainer.is_global_zero}")
        if self.trainer.is_global_zero:  # 메인 프로세스에서만 저장
            print(f"[DEBUG] Calling save_temperature_values...")
            self.save_temperature_values()
        else:
            print(f"[DEBUG] Skipping temperature save - not global zero")

        # ✅ 메모리 누수 방지: 플래그 리셋
        self.has_detection = False
        self.has_pixel = False

def main(args):
    # ✅ DDP 환경변수 설정 (deadlock 방지)
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # ✅ NCCL 설정 (교착상태 방지)
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30분
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    
    # ✅ DataLoader와 OpenCV의 충돌로 인한 교착 상태(deadlock)를 방지합니다.
    #    num_workers > 0 일 때, OpenCV가 내부적으로 멀티스레딩을 사용하면
    #    PyTorch의 멀티프로세싱과 충돌하여 프로그램이 멈출 수 있습니다.
    cv2.setNumThreads(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    pl.seed_everything(args.seed, workers=True)  # ✅ 학습 reproducibility 보장
    if args.deterministic in ("full", "not_pl"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cfg = get_cfg(args)
    
    # Stage 1 모델 경로 자동 설정
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        # checkpoint가 비어있으면 기본 경로에서 best 모델 찾기
        if hasattr(cfg.train, 'stage') and cfg.train.stage == 1:
            # Stage 1인 경우 best_s1.pth 사용
            checkpoint_path = os.path.join(cfg.commen.model_dir, 'best_s1.pth')
        else:
            # Stage 2인 경우 기본 best.pth 사용
            checkpoint_path = os.path.join(cfg.commen.model_dir, 'best.pth')
        print(f"[INFO] Using default checkpoint: {checkpoint_path}")
    
    model = LightningTester(cfg, checkpoint_path)
    data_loader = make_data_loader(is_train=False, cfg=cfg)

    # ✅ GPU 전략 결정
    num_gpus = torch.cuda.device_count()
    if args.gpu_strategy == 'ddp' and num_gpus > 1:
        # ✅ DDP 교착상태 방지를 위한 추가 설정
        from pytorch_lightning.strategies import DDPStrategy
        from datetime import timedelta
        strategy = DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(minutes=2),  # 2분 타임아웃
            process_group_backend="nccl",
        )
        print(f"✅ Using DDP strategy for test with {num_gpus} GPUs")
        devices = num_gpus
    elif args.gpu_strategy == 'dp' and num_gpus > 1:
        strategy = "dp"
        print(f"✅ Using DataParallel (DP) for test with {num_gpus} GPUs")
        devices = num_gpus
    else:
        strategy = "auto"  # single GPU거나 자동
        print(f"✅ Using single GPU strategy for test")
        devices = 1
        
    trainer = pl.Trainer(accelerator="gpu",
                         devices=devices,
                         strategy=strategy,
                         logger=False,
                         precision=args.precision,
                         deterministic=True if args.deterministic == 'full' else False,
                         replace_sampler_ddp=False, # ✅ Lightning 자동 DistributedSampler 비활성화
                         )

    # ✅ dataloaders 인자를 제거합니다.
    #    Trainer가 모델에 정의된 `test_dataloader()` 메서드를 자동으로 호출합니다.
    trainer.test(model, dataloaders=data_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dataset", default="None")
    parser.add_argument("--with_nms", default=False, action='store_true')
    parser.add_argument("--nms_iou_th", default=0.5, type=float, help="IoU threshold for NMS")
    parser.add_argument("--nms_containment_th", default=0.7, type=float, help="Containment threshold for NMS")
    parser.add_argument("--eval", default='segm', choices=['segm', 'bbox'])
    parser.add_argument("--stage", default='final', choices=['init', 'coarse', 'final'])
    parser.add_argument("--type", default='accuracy', choices=['speed', 'accuracy'])
    # parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--exp", default="None")
    parser.add_argument("--with_deform_metric", action='store_false')
    parser.add_argument("--th_iou", default=0.99, type=float)
    parser.add_argument("--print_only_deform_metric", action='store_true', default=False)
    parser.add_argument("--save_featuremap", action='store_true', default=False)
    parser.add_argument("--check_simple", action='store_false', default=True)
    parser.add_argument("--not_use_vertex_reduction", action='store_true', default=False)
    parser.add_argument("--th_score_vertex_cls", default=0.6, type=float)
    parser.add_argument("--reduce_apply_adaptive", action='store_true', default=False)
    parser.add_argument("--rotate_tta", action='store_true', default=False)
    parser.add_argument("--single_rotate_angle", default=None, type=int)
    parser.add_argument("--gpu_strategy", default="dp")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--precision", default=32, choices=[32, 16, 64], type=int)
    parser.add_argument("--num_workers", default=None, type=int, help="Number of workers for data loading")
    parser.add_argument("--viz", action='store_true', help="Visualize all test results")
    parser.add_argument("--viz_mode", choices=["final", "timeline"], default="final",
                        help="Visualization mode: 'final' saves only the last stage; 'timeline' saves [poly_init, poly_coarse, py[0]..py[-1]] in a horizontal strip.")
    parser.add_argument("--test_bs", default=None, type=int, help="Test batch size. It will be divided by the number of GPUs.")
    parser.add_argument("--vis_ct_score", default=0.2, type=float, help="Confidence threshold for visualization")
    parser.add_argument("--simple_ratio_cts", nargs='+', type=float, default=[0.1, 0.2, 0.3], help="List of confidence thresholds for simple ratio calculation")
    parser.add_argument("--deterministic", default="full", choices=["full", "not_pl", "never"])
    parser.add_argument("--track_self_intersection", action='store_true', default=False)
    parser.add_argument("--ct_score", default=0.05, type=float, help="Confidence threshold for test")
    parser.add_argument("--viz_pixel_ct", action='store_true', default=False, help="Visualize pixel head and ct_hm spatial alignment during test")
    parser.add_argument("--viz_pixel_initial_contours", action='store_true', default=False, help="Visualize pixel map initial contours before ct_score filtering")
    parser.add_argument("--viz_pixel_maps", action='store_true', default=False, help="Visualize pixel maps")
    parser.add_argument("--viz_pixel_mode", choices=['final', 'all'], default='final', help="Pixel map visualization mode: 'final' (last stage only) or 'all' (all stages)")
    parser.add_argument("--viz_pixel_raw", action='store_true', default=False, help="Visualize unnormalized raw pixel maps (before softmax/sigmoid) with colorbar")
    parser.add_argument("--viz_snake_energy", action='store_true', default=False, help="Calculate and display snake energy values on each contour")
    parser.add_argument("--with_vertex_reordering", action='store_true', default=False, help="Apply vertex re-ordering post-processing to reduce self-intersections")
    parser.add_argument("--vertex_reorder_method", choices=['auto', 'local_triangle', 'angle_based', 'local_swap'], default='auto', help="Vertex reordering method")
    parser.add_argument("--viz_stage1_init", action='store_true', default=False, help="Visualize poly_init in Stage 1 when ct_hm or wh heads are trained")
    parser.add_argument("--viz_self_intersection", action='store_true', default=False, help="Visualize contours with self-intersections, highlighting intersecting segments")
    parser.add_argument("--viz_si_save_all", action='store_true', default=False, help="Save all images for self-intersection visualization, even without intersections (for debugging)")
    parser.add_argument("--log_si_to_csv", action='store_true', default=False, help="Log self-intersecting contours to CSV file with image name and bbox info")
    parser.add_argument("--viz_line_thickness", default=2, type=int, help="Line thickness for contour visualization (default: 2)")
    args = parser.parse_args()
    main(args)
