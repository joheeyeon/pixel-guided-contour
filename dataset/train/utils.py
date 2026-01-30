import numpy as np
import cv2
import random
from shapely.geometry import Polygon
import shapely

# for deepsnake
def get_extreme_points(pts):
    l, t = min(pts[:, 0]), min(pts[:, 1])
    r, b = max(pts[:, 0]), max(pts[:, 1])
    # 3 degrees
    thresh = 0.02
    w = r - l + 1
    h = b - t + 1

    # t_idx = np.argmin(pts[:, 1])
    #rev30>>
    t_val = np.min(pts[:, 1])
    t_inds = np.nonzero(pts[:, 1] == t_val)
    t_center_ind = np.argsort(pts[t_inds[0],0])[int(len(t_inds[0])*0.5)]
    t_idx = t_inds[0][t_center_ind]
    #<<rev30
    t_idxs = [t_idx]
    tmp = (t_idx + 1) % pts.shape[0]
    while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
        t_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (t_idx - 1) % pts.shape[0]
    while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
        t_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    tt = [(max(pts[t_idxs, 0]) + min(pts[t_idxs, 0])) / 2, t]

    # b_idx = np.argmax(pts[:, 1])
    # rev30>>
    b_val = np.max(pts[:, 1])
    b_inds = np.nonzero(pts[:, 1] == b_val)
    b_center_ind = np.argsort(pts[b_inds[0], 0])[int(len(b_inds[0]) * 0.5)]
    b_idx = b_inds[0][b_center_ind]
    # <<rev30
    b_idxs = [b_idx]
    tmp = (b_idx + 1) % pts.shape[0]
    while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
        b_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (b_idx - 1) % pts.shape[0]
    while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
        b_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    bb = [(max(pts[b_idxs, 0]) + min(pts[b_idxs, 0])) / 2, b]

    # l_idx = np.argmin(pts[:, 0])
    # rev30>>
    l_val = np.min(pts[:, 0])
    l_inds = np.nonzero(pts[:, 0] == l_val)
    l_center_ind = np.argsort(pts[l_inds[0], 1])[int(len(l_inds[0]) * 0.5)]
    l_idx = l_inds[0][l_center_ind]
    # <<rev30
    l_idxs = [l_idx]
    tmp = (l_idx + 1) % pts.shape[0]
    while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
        l_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (l_idx - 1) % pts.shape[0]
    while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
        l_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    ll = [l, (max(pts[l_idxs, 1]) + min(pts[l_idxs, 1])) / 2]

    # r_idx = np.argmax(pts[:, 0])
    # rev30>>
    r_val = np.max(pts[:, 0])
    r_inds = np.nonzero(pts[:, 0] == r_val)
    r_center_ind = np.argsort(pts[r_inds[0], 1])[int(len(r_inds[0]) * 0.5)]
    r_idx = r_inds[0][r_center_ind]
    # <<rev30
    r_idxs = [r_idx]
    tmp = (r_idx + 1) % pts.shape[0]
    while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
        r_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (r_idx - 1) % pts.shape[0]
    while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
        r_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    rr = [r, (max(pts[r_idxs, 1]) + min(pts[r_idxs, 1])) / 2]

    return np.array([tt, ll, bb, rr])

def get_octagon(ex):
    w, h = ex[3][0] - ex[1][0], ex[2][1] - ex[0][1]
    t, l, b, r = ex[0][1], ex[1][0], ex[2][1], ex[3][0]
    x = 8.0
    octagon = [
        ex[0][0], ex[0][1],
        max(ex[0][0] - w / x, l), ex[0][1],
        ex[1][0], max(ex[1][1] - h / x, t),
        ex[1][0], ex[1][1],
        ex[1][0], min(ex[1][1] + h / x, b),
        max(ex[2][0] - w / x, l), ex[2][1],
        ex[2][0], ex[2][1],
        min(ex[2][0] + w / x, r), ex[2][1],
        ex[3][0], min(ex[3][1] + h / x, b),
        ex[3][0], ex[3][1],
        ex[3][0], max(ex[3][1] - h / x, t),
        min(ex[0][0] + w / x, r), ex[0][1],
    ]
    return np.array(octagon).reshape(-1, 2)

def get_quadrangle(box):
    x_min, y_min, x_max, y_max = box
    quadrangle = [
        [(x_min + x_max) / 2., y_min],
        [x_min, (y_min + y_max) / 2.],
        [(x_min + x_max) / 2., y_max],
        [x_max, (y_min + y_max) / 2.]
    ]
    return np.array(quadrangle)


def get_box(box):
    x_min, y_min, x_max, y_max = box
    box = [
        [(x_min + x_max) / 2., y_min],
        [x_min, y_min],
        [x_min, (y_min + y_max) / 2.],
        [x_min, y_max],
        [(x_min + x_max) / 2., y_max],
        [x_max, y_max],
        [x_max, (y_min + y_max) / 2.],
        [x_max, y_min]
    ]
    return np.array(box)


def get_init(box, init='quadrangle'):
    if init == 'quadrangle':
        return get_quadrangle(box)
    else:
        return get_box(box)
def make_simply_connected(polygon_coords):
    """
    Modify the given polygon to make it simply connected by removing holes and fixing intersections.

    Args:
        polygon_coords (list of tuples): List of (x, y) coordinates representing the polygon.

    Returns:
        Polygon: A simply connected polygon.
    """
    def process_except(polygon):
        if polygon.buffer(0).geom_type == 'MultiPolygon':
            poly_list = []
            for poly in polygon.buffer(0).geoms:
                if (not poly.is_empty) and (poly.is_simple) and (round(poly.area,2) > 0):
                    poly_list.append(np.array(list(poly.exterior.coords))[:-1,:])
            return poly_list
        elif polygon.buffer(0).is_empty:
            return []
        elif polygon.buffer(0).geom_type == 'Polygon':
            return np.array(list(polygon.buffer(0).exterior.coords))[:-1,:]
        else:
            return []
    # Create a Shapely Polygon object
    polygon = Polygon(polygon_coords)

    # Fix self-intersections if present
    if not polygon.is_valid:
        forward_list = process_except(polygon)
        if len(forward_list) > 0:
            return forward_list
        elif Polygon(polygon_coords[::-1, :]).is_valid:
            polygon = Polygon(polygon_coords[::-1, :])
        else:
            return process_except(Polygon(polygon_coords[::-1, :]))
        # rev17
        # if shapely.is_empty(polygon.buffer(0)) or shapely.get_type_id(polygon) == 6 or shapely.get_type_id(polygon.buffer(0)) == 6:
        #     return []
        # else:
        #     polygon = polygon.buffer(0)
        # f"is not valid : {polygon.buffer(0)}")
        # polygon = polygon.buffer(0)
        # return []

    # Extract the exterior and any holes
    exterior = polygon.exterior

    # Create a new Polygon without holes
    new_polygon = np.array(list(exterior.coords))

    return new_polygon[:-1,:]

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
                         inv=0,
                         shear=np.array([0, 0], dtype=np.float32)):
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

    trans[0, 1] += shear[0]
    trans[1, 0] += shear[1]
    return trans


def affine_transform(pt, t):
    """pt: [n, 2]"""
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt


def get_border(border, size):
    i = 1
    while np.any(size - border // i <= border // i):
        i *= 2
    return border // i

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)
    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)

def zoom(img, poly, margin=10):
    y_min, y_max = np.min(poly, axis=0), np.max(poly, axis=0)
    x_min, x_max = np.min(poly, axis=1), np.max(poly, axis=1)
    init_ind_box = (x_min-margin, y_min-margin)


    return init_ind_box

def augment(img, split, _data_rng, _eig_val, _eig_vec, mean, std,
            down_ratio, input_h, input_w, scale_range, scale=None, test_rescale=None, test_scale=None, flip_type='lr',
            list_curved_py=np.array([0]), img_gt=None, is_shift=True, gt_mask_labels=None, rot_type=None, rot_angle=0):
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    #scale : default size of image
    if scale is None:
        scale = max(img.shape[0], img.shape[1]) * 1.0
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    flipped = False
    # rot_angle = 0
    if split == 'train':
        scale = scale * (random.random() * (scale_range[1] - scale_range[0]) + scale_range[0]) #scale_range : the range for scaling
        x, y = center
        if is_shift:
            w_border = get_border(width/4, scale[0]) + 1
            h_border = get_border(height/4, scale[1]) + 1
            center[0] = np.random.randint(low=max(x - w_border, 0), high=min(x + w_border, width - 1))
            center[1] = np.random.randint(low=max(y - h_border, 0), high=min(y + h_border, height - 1))

        if flip_type == 'lr':
            if random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                if img_gt is not None:
                    img_gt = img_gt[:, ::-1]
                center[0] = width - center[0] - 1
        else:
            flip_rand = random.random()
            if flip_rand < 0.25:
                flipped = 'ud'
                img = img[::-1, :, :]
                if img_gt is not None:
                    img_gt = img_gt[::-1, :]
                center[1] = height - center[1] - 1
            elif flip_rand < 0.5:
                flipped = 'lr'
                img = img[:, ::-1, :]
                if img_gt is not None:
                    img_gt = img_gt[:, ::-1]
                center[0] = width - center[0] - 1
            elif flip_rand < 0.75:
                flipped = 'udlr'
                img = img[::-1, ::-1, :]
                if img_gt is not None:
                    img_gt = img_gt[::-1, ::-1]
                center[0] = width - center[0] - 1
                center[1] = height - center[1] - 1

        if rot_type == 'random':
            rot_rand = random.random()
            if rot_rand >= 0.25:
                rot_angle = 90
            elif rot_rand >= 0.5:
                rot_angle = 180
            elif rot_rand >= 0.75:
                rot_angle = 270

        if rot_angle != 0:
            img = np.rot90(img, rot_angle // 90)
            if img_gt is not None:
                img_gt = np.rot90(img_gt, rot_angle // 90)

    if split != 'train':
        scale = np.array([width, height])
        x = 32
        if test_rescale is not None:
            input_w, input_h = int((width / test_rescale + x - 1) // x * x),\
                               int((height / test_rescale + x - 1) // x * x)
        else:
            if test_scale is None:
                input_w = (int(width / 1.) | (x - 1)) + 1
                input_h = (int(height / 1.) | (x - 1)) + 1
            else:
                scale = max(width, height) * 1.0
                scale = np.array([scale, scale])
                input_w, input_h = test_scale
        center = np.array([width // 2, height // 2])

    if (split == 'train') & (list_curved_py.sum() > 0):
        rot = random.uniform(0, 359)
        shear_x = random.uniform(-0.5, 0.5)
        shear_y = random.uniform(-0.5, 0.5)
        shear = np.array([shear_x, shear_y], dtype=np.float32)
    else:
        rot = 0
        shear = np.array([0., 0.], dtype=np.float32)

    trans_input = get_affine_transform(center, scale, rot, [input_w, input_h], shear=shear) #input_w, input_h : size of input to network
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    if img_gt is not None:
        inp_gt = cv2.resize(img_gt, (width, height), interpolation=cv2.INTER_NEAREST)
        inp_gt = cv2.warpAffine(inp_gt, trans_input, (input_w, input_h), flags=cv2.INTER_NEAREST)
    else:
        inp_gt = None

    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        color_aug(_data_rng, inp, _eig_val, _eig_vec)

    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    output_h, output_w = input_h // down_ratio, input_w // down_ratio
    trans_output = get_affine_transform(center, scale, rot, [output_w, output_h], shear=shear)
    inp_out_hw = (input_h, input_w, output_h, output_w)

    if (gt_mask_labels is not None) and (inp_gt is not None):
        for k in gt_mask_labels.keys():
            inp_gt[inp_gt == k] = gt_mask_labels[k]

    return orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw, inp_gt, rot_angle

def handle_break_point(poly, axis, number, outside_border):
    if len(poly) == 0:
        return []

    if len(poly[outside_border(poly[:, axis], number)]) == len(poly):
        return []

    break_points = np.argwhere(
        outside_border(poly[:-1, axis], number) != outside_border(poly[1:, axis], number)).ravel()
    if len(break_points) == 0:
        return poly

    new_poly = []
    if not outside_border(poly[break_points[0], axis], number):
        new_poly.append(poly[:break_points[0]])

    for i in range(len(break_points)):
        current_poly = poly[break_points[i]]
        next_poly = poly[break_points[i] + 1]
        mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (next_poly[axis] - current_poly[axis])

        if outside_border(poly[break_points[i], axis], number):
            if mid_poly[axis] != next_poly[axis]:
                new_poly.append([mid_poly])
            next_point = len(poly) if i == (len(break_points) - 1) else break_points[i + 1]
            new_poly.append(poly[break_points[i] + 1:next_point])
        else:
            new_poly.append([poly[break_points[i]]])
            if mid_poly[axis] != current_poly[axis]:
                new_poly.append([mid_poly])

    if outside_border(poly[-1, axis], number) != outside_border(poly[0, axis], number):
        current_poly = poly[-1]
        next_poly = poly[0]
        mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (next_poly[axis] - current_poly[axis])
        new_poly.append([mid_poly])

    return np.concatenate(new_poly)

def transform_polys(polys, trans_output, output_h, output_w):
    new_polys = []
    for i in range(len(polys)):
        poly = polys[i]
        poly = affine_transform(poly, trans_output)
        poly = handle_break_point(poly, 0, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
        poly = handle_break_point(poly, 1, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
        if len(poly) == 0:
            continue
        if len(np.unique(poly, axis=0)) <= 2:
            continue
        new_polys.append(poly)
    return new_polys

def filter_tiny_polys(polys):
    return [poly for poly in polys if Polygon(poly).area > 5]

def get_cw_polys(polys):
    return [poly[::-1] if Polygon(poly).exterior.is_ccw else poly for poly in polys]

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    if b3 ** 2 - 4 * a3 * c3 < 0:
        r3 = min(r1, r2)
    else:
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp

def four_idx(img_gt_poly):
    x_min, y_min = np.min(img_gt_poly, axis=0)
    x_max, y_max = np.max(img_gt_poly, axis=0)
    center = [(x_min + x_max) / 2., (y_min + y_max) / 2.]
    can_gt_polys = img_gt_poly.copy()
    can_gt_polys[:, 0] -= center[0]
    can_gt_polys[:, 1] -= center[1]
    distance = np.sum(can_gt_polys ** 2, axis=1, keepdims=True) ** 0.5 + 1e-6
    can_gt_polys /= np.repeat(distance, axis=1, repeats=2)
    idx_bottom = np.argmax(can_gt_polys[:, 1])
    idx_top = np.argmin(can_gt_polys[:, 1])
    idx_right = np.argmax(can_gt_polys[:, 0])
    idx_left = np.argmin(can_gt_polys[:, 0])
    return [idx_bottom, idx_right, idx_top, idx_left]

# def get_img_gt(img_gt_poly, idx, t=128):
#     align = len(idx)
#     pointsNum = img_gt_poly.shape[0]
#     r = []
#     k = np.arange(0, t / align, dtype=float) / (t / align)
#     for i in range(align):
#         begin = idx[i]
#         end = idx[(i + 1) % align]
#         if begin > end:
#             end += pointsNum
#         r.append((np.round(((end - begin) * k).astype(int)) + begin) % pointsNum)
#     r = np.concatenate(r, axis=0)
#     return img_gt_poly[r, :]
def get_img_gt(img_gt_poly, idx, t=128):#rev20
    align = len(idx)
    pointsNum = img_gt_poly.shape[0]
    r = []
    k = np.linspace(0, 1, int(t // align), endpoint=False)  # Uniformly distribute the points

    for i in range(align):
        begin = idx[i]
        end = idx[(i + 1) % align]
        if begin > end:
            end += pointsNum

        # Sample points between begin and end
        sampled_points = np.round((begin + (end - begin) * k)).astype(int) % pointsNum
        r.extend(sampled_points)

    start_ind = r[0]
    r = np.sort(np.array(r))
    shift_ind = np.argwhere(r == start_ind)
    r = np.roll(r, -shift_ind[0])

    # Get the points based on the sampled indices
    sampled_points = img_gt_poly[r, :]
    return sampled_points

def img_poly_to_can_poly(img_poly):
    x_min, y_min = np.min(img_poly, axis=0)
    can_poly = img_poly - np.array([x_min, y_min])
    return can_poly
