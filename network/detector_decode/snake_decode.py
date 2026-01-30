import torch
from .utils import decode_ct_hm, clip_to_image, get_gcn_feature, uniform_upsample

def get_init(box, init='quadrangle'):
    if init == 'quadrangle':
        return get_quadrangle(box)
    else:
        return get_box(box)

def get_quadrangle(box):
    x_min, y_min, x_max, y_max = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    quadrangle = [
        (x_min + x_max) / 2., y_min,
        x_min, (y_min + y_max) / 2.,
        (x_min + x_max) / 2., y_max,
        x_max, (y_min + y_max) / 2.
    ]
    quadrangle = torch.stack(quadrangle, dim=2).view(x_min.size(0), x_min.size(1), 4, 2)
    return quadrangle

def get_box(box):
    x_min, y_min, x_max, y_max = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    box = [
        (x_min + x_max) / 2., y_min,
        x_min, y_min,
        x_min, (y_min + y_max) / 2.,
        x_min, y_max,
        (x_min + x_max) / 2., y_max,
        x_max, y_max,
        x_max, (y_min + y_max) / 2.,
        x_max, y_min
    ]
    box = torch.stack(box, dim=2).view(x_min.size(0), x_min.size(1), 8, 2)
    return box

def get_octagon(ex):
    w, h = ex[..., 3, 0] - ex[..., 1, 0], ex[..., 2, 1] - ex[..., 0, 1]
    t, l, b, r = ex[..., 0, 1], ex[..., 1, 0], ex[..., 2, 1], ex[..., 3, 0]
    x = 8.

    octagon = [
        ex[..., 0, 0], ex[..., 0, 1],
        torch.max(ex[..., 0, 0] - w / x, l), ex[..., 0, 1],
        ex[..., 1, 0], torch.max(ex[..., 1, 1] - h / x, t),
        ex[..., 1, 0], ex[..., 1, 1],
        ex[..., 1, 0], torch.min(ex[..., 1, 1] + h / x, b),
        torch.max(ex[..., 2, 0] - w / x, l), ex[..., 2, 1],
        ex[..., 2, 0], ex[..., 2, 1],
        torch.min(ex[..., 2, 0] + w / x, r), ex[..., 2, 1],
        ex[..., 3, 0], torch.min(ex[..., 3, 1] + h / x, b),
        ex[..., 3, 0], ex[..., 3, 1],
        ex[..., 3, 0], torch.max(ex[..., 3, 1] - h / x, t),
        torch.min(ex[..., 0, 0] + w / x, r), ex[..., 0, 1]
    ]
    octagon = torch.stack(octagon, dim=2).view(t.size(0), t.size(1), 12, 2)

    return octagon

