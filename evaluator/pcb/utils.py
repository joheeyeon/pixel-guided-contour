import pycocotools.mask as mask_utils
import numpy as np
import cv2

def coco_poly_to_rle(poly, h, w):
    rle_ = []
    for i in range(len(poly)):
        # Convert tensor/numpy and normalize format
        try:
            # 1. Convert torch tensor to numpy
            if hasattr(poly[i], 'detach'):  # torch.Tensor
                poly_np = poly[i].detach().cpu().numpy()
            else:
                poly_np = np.asarray(poly[i])
            
            # 2. Clean up dimensions (3D -> 2D)
            while poly_np.ndim > 2:
                if poly_np.shape[0] == 1:
                    poly_np = poly_np[0]
                elif poly_np.shape[-1] == 2:
                    poly_np = poly_np.reshape(-1, 2)
                    break
                else:
                    poly_np = poly_np[0]
            
            # 3. Final validation and flatten
            if poly_np.ndim == 2 and poly_np.shape[1] == 2:
                coords = poly_np.reshape(-1).astype(np.float64)  # pycocotools compatible
            elif poly_np.ndim == 1:
                coords = poly_np.astype(np.float64)
            else:
                print(f"[WARN] Invalid polygon shape: {poly_np.shape}, skipping")
                continue
            
            # 4. Check minimum requirement (6 coords = 3 points)
            if len(coords) < 6:
                print(f"[WARN/evaluator] Polygon has less than 3 points ({len(coords)//2}), skipping")
                continue
            
            # 5. Call pycocotools
            rles = mask_utils.frPyObjects([coords.tolist()], h, w)
            rle = mask_utils.merge(rles)
            rle['counts'] = rle['counts'].decode('utf-8')
            rle_.append(rle)
            
        except Exception as e:
            print(f"[WARN] Failed to process polygon {i}: {e}")
            continue
    
    return rle_

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    """pt: [n, 2]"""
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt


