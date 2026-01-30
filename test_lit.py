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
    cfg.test.visualize = args.viz
    cfg.test.viz_mode = args.viz_mode
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

    # ✅ 테스트 시에도 num_workers를 CLI 인자로 설정할 수 있도록 추가합니다.
    if args.num_workers is not None:
        cfg.test.num_workers = args.num_workers
    # ✅ 테스트 배치 사이즈를 CLI 인자로 받아 GPU 수만큼 나눠서 설정합니다.
    if args.test_bs is not None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        cfg.test.batch_size = int(int(args.test_bs) / num_gpus)

    # ✅ config 파일의 기존 설정을 CLI 인자로 덮어쓰지 않고 OR 조건으로 처리
    # ✅ Pixel map visualization 설정 추가
    # ✅ Vertex reordering 설정 추가
    
    # ✅ Vertex reordering 설정을 경로에 반영 (설정 완료 후)
    
    # ✅ Stage 1 poly_init visualization 설정 추가
    # ✅ Self-intersection visualization 설정 추가
    # ✅ Self-intersection CSV logging 옵션 추가
    # ✅ Contour 시각화 두께 설정

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
        
        self.evaluator = make_evaluator(cfg)
        self.evaluator_b = make_evaluator(cfg, format='bound')
        self.evaluator_pix = SegmentationMetrics()
            
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
            output = self.network(inp, batch=batch)
        
        #------- edit:feat:self-intersection-count:25-08-10 ---------
        # ✅ 메모리 누수 방지: 전체 output 저장 대신 플래그만 설정
        if 'detection' in output:
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
    
    def on_test_epoch_end(self):
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
    parser.add_argument("--gpu_strategy", default="dp")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=None, type=int, help="Number of workers for data loading")
    parser.add_argument("--precision", default=32, choices=[32, 16, 64], type=int)
    parser.add_argument("--viz", action='store_true', help="Visualize all test results")
    parser.add_argument("--viz_mode", choices=["final", "timeline"], default="final",
                        help="Visualization mode: 'final' saves only the last stage; 'timeline' saves [poly_init, poly_coarse, py[0]..py[-1]] in a horizontal strip.")
    parser.add_argument("--test_bs", default=None, type=int, help="Test batch size. It will be divided by the number of GPUs.")
    parser.add_argument("--deterministic", default="full", choices=["full", "not_pl", "never"])
    parser.add_argument("--ct_score", default=0.05, type=float, help="Confidence threshold for test")
    args = parser.parse_args()
    main(args)
