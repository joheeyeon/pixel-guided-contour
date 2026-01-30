import rdp
import math, shapely
import numpy as np
import os, pickle
import torch.utils.data as data
from pycocotools.coco import COCO
from .douglas import Douglas
from .utils import transform_polys, filter_tiny_polys, get_cw_polys, gaussian_radius, draw_umich_gaussian,\
uniformsample, four_idx, get_img_gt, img_poly_to_can_poly, augment, make_simply_connected, get_init, get_octagon, get_extreme_points

class Dataset(data.Dataset):
    def __init__(self, anno_file, data_root, split, cfg, data_root_gt=None):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.data_root = data_root
        self.data_root_gt = data_root_gt
        self.split = split

        self.coco = COCO(anno_file)
        self.anns = np.array(sorted(self.coco.getImgIds()))
        print(split)
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        print(len(self.anns))
        self.json_category_id_to_continuous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.d = Douglas(num_vertices=self.cfg.train.iou_params['num_keypoints'] if 'num_keypoints' in self.cfg.train.iou_params else None,
                         D=cfg.data.douglas['D'] if 'D' in cfg.data.douglas else 3)
        # self.tmp_c = 0

    def transform_original_data(self, instance_polys, flipped, height, width, trans_output, inp_out_hw, rot_angle=0):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]
            if (isinstance(flipped, bool) and flipped) or flipped == 'lr':
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_
            elif flipped == 'ud':
                polys_ = []
                for poly in polys:
                    poly[:, 1] = height - np.array(poly[:, 1]) - 1
                    polys_.append(poly.copy())
                polys = polys_
            elif flipped == 'udlr':
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    poly[:, 1] = height - np.array(poly[:, 1]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            if rot_angle != 0:
                cx = (width - 1) / 2
                cy = (height - 1) / 2
                polys_ = []
                for poly in polys:
                    x = poly[:, 0] - cx
                    y = poly[:, 1] - cy
                    if rot_angle%360 == 90:
                        x_new = y
                        y_new = -x
                    elif rot_angle%360 == 180:
                        x_new = -x
                        y_new = -y
                    elif rot_angle%360 == 270:
                        x_new = -y
                        y_new = x

                    poly[:, 0] = x_new + cx
                    poly[:, 1] = y_new + cy
                    polys_.append(poly.copy())
                polys = polys_

            polys = transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def check_curved(self, anno, instance_polys, rdp_eps, rdp_th_n_pts):
        blist_curved = []
        for i in range(len(anno)):
            instance_poly = instance_polys[i]

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                poly_simple = rdp.rdp(poly, rdp_eps)
                if poly_simple.shape[0] > rdp_th_n_pts:
                    blist_curved.append(1)
                else:
                    blist_curved.append(0)
        return blist_curved

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = filter_tiny_polys(instance)
            polys = get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    # for deep snake
    def prepare_init(self, box, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = get_init(box)
        img_init_poly = uniformsample(img_init_poly, self.cfg.commen.init_points_per_poly)
        can_init_poly = img_poly_to_can_poly(img_init_poly)
        img_gt_poly = extreme_point
        can_gt_poly = img_poly_to_can_poly(img_gt_poly)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_snake_evolution(self, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        octagon = get_octagon(extreme_point)
        img_init_poly = uniformsample(octagon, self.cfg.data.points_per_poly)
        can_init_poly = img_poly_to_can_poly(img_init_poly)

        img_gt_poly = uniformsample(poly, len(poly) * self.cfg.data.points_per_poly)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = img_poly_to_can_poly(img_gt_poly)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)


    def prepare_evolution(self, poly, img_gt_polys, can_gt_polys, keyPointsMask, img_gt_init_polys, can_gt_init_polys, img_gt_coarse_polys, can_gt_coarse_polys):
        num_init = self.cfg.commen.init_points_per_poly if 'wh' not in self.cfg.model.heads else self.cfg.model.heads['wh']/2
        img_gt_init_poly = uniformsample(poly, len(poly) * num_init)
        img_gt_coarse_poly = uniformsample(poly, len(poly) * self.cfg.commen.init_points_per_poly)
        img_gt_poly = uniformsample(poly, len(poly) * self.cfg.data.points_per_poly)
        #rev19 recover
        idx = four_idx(img_gt_poly)
        init_idx = four_idx(img_gt_init_poly)
        coarse_idx = four_idx(img_gt_coarse_poly)
        img_gt_poly = get_img_gt(img_gt_poly, idx, t=self.cfg.data.points_per_poly)
        img_gt_init_poly = get_img_gt(img_gt_init_poly, init_idx, t=num_init)
        img_gt_coarse_poly = get_img_gt(img_gt_coarse_poly, coarse_idx, t=self.cfg.commen.init_points_per_poly)
        # rev18
        # space = len(img_gt_poly) / self.cfg.data.points_per_poly
        # img_gt_poly = img_gt_poly[np.arange(0, len(img_gt_poly), space).round().astype(np.int32)]
        # space_init = len(img_gt_init_poly) / (self.cfg.model.heads['wh']/2)
        # img_gt_init_poly = img_gt_init_poly[np.arange(0, len(img_gt_init_poly), space_init).round().astype(np.int32)]
        # space_coarse = len(img_gt_coarse_poly) / self.cfg.commen.init_points_per_poly
        # img_gt_coarse_poly = img_gt_coarse_poly[np.arange(0, len(img_gt_coarse_poly), space_coarse).round().astype(np.int32)]

        can_gt_poly = img_poly_to_can_poly(img_gt_poly)
        can_gt_init_poly = img_poly_to_can_poly(img_gt_init_poly)
        can_gt_coarse_poly = img_poly_to_can_poly(img_gt_coarse_poly)
        if self.cfg.data.get_keypoints_mask:
            key_mask = self.get_keypoints_mask(img_gt_poly)
            keyPointsMask.append(key_mask)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)
        img_gt_init_polys.append(img_gt_init_poly)
        can_gt_init_polys.append(can_gt_init_poly)
        img_gt_coarse_polys.append(img_gt_coarse_poly)
        can_gt_coarse_polys.append(can_gt_coarse_poly)

    # def prepare_polyrnn(self, img, polygon):
    #     # zoom the region around polygon on img
    #     point_count = 2
    #     #         img_array = np.zeros([data_num, 3, 224, 224])
    #     label_array = np.zeros([self.length, 28 * 28 + 3])
    #     label_index_array = np.zeros([self.length])
    #     if point_num < self.length - 3:
    #         for points in polygon:
    #             index_a = int(points[0] / 8)
    #             index_b = int(points[1] / 8)
    #             index = index_b * 28 + index_a
    #             label_array[point_count, index] = 1
    #             label_index_array[point_count] = index
    #             point_count += 1
    #         label_array[point_count, 28 * 28] = 1
    #         label_index_array[point_count] = 28 * 28
    #         for kkk in range(point_count + 1, self.length):
    #             if kkk % (point_num + 3) == point_num + 2:
    #                 index = 28 * 28
    #             elif kkk % (point_num + 3) == 0:
    #                 index = 28 * 28 + 1
    #             elif kkk % (point_num + 3) == 1:
    #                 index = 28 * 28 + 2
    #             else:
    #                 index_a = int(polygon[kkk % (point_num + 3) - 2][0] / 8)
    #                 index_b = int(polygon[kkk % (point_num + 3) - 2][1] / 8)
    #                 index = index_b * 28 + index_a
    #             label_array[kkk, index] = 1
    #             label_index_array[kkk] = index
    #     else:
    #         scale = point_num * 1.0 / (self.length - 3)
    #         index_list = (np.arange(0, self.length - 3) * scale).astype(int)
    #         for points in polygon[index_list]:
    #             index_a = int(points[0] / 8)
    #             index_b = int(points[1] / 8)
    #             index = index_b * 28 + index_a
    #             label_array[point_count, index] = 1
    #             label_index_array[point_count] = index
    #             point_count += 1
    #         for kkk in range(point_count, self.length):
    #             index = 28 * 28
    #             label_array[kkk, index] = 1
    #             label_index_array[kkk] = index
    #
    #     return zoom_img, zoom_polys, init_ind_box, label_array, label_index_array

    def get_keypoints_mask(self, img_gt_poly):
        key_mask = self.d.sample(img_gt_poly)
        # from scipy.io import savemat
        # import os
        # os.makedirs(f"{self.cfg.commen.result_dir}/OnTraining/Polyloss", exist_ok=True)
        # savemat(f"{self.cfg.commen.result_dir}/OnTraining/Polyloss/douglas_{self.tmp_c}.mat",
        #         {'poly': img_gt_poly,
        #          'mask': key_mask})
        # self.tmp_c += 1
        return key_mask

    def __getitem__(self, base_index):
        if self.cfg.data.augment_rotate == 'all_4':
            index = base_index // 4
            rot_idx = base_index % 4
            fixed_rot_angle = [0, 90, 180, 270][rot_idx]
        else:
            index = base_index
            fixed_rot_angle = 0

        data_input = {}
        ann = self.anns[index]
        anno, image_path, image_id = self.process_info(ann)
        img, instance_polys, cls_ids = self.read_original_data(anno, image_path)
        gt_img = self.get_gt_img(ann)
        width, height = img.shape[1], img.shape[0]
        if self.cfg.data.add_augment_curved:
            list_curved_py = self.check_curved(anno, instance_polys, self.cfg.data.rdp_eps, self.cfg.data.rdp_th_n_pts)
        else:
            list_curved_py = np.array([0])
        # print(f"{image_path} :: {list_curved_py}")
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw, inp_gt, rot_angle = \
            augment(
                img, self.split,
                self.cfg.data.data_rng, self.cfg.data.eig_val, self.cfg.data.eig_vec,
                self.cfg.data.mean, self.cfg.data.std, self.cfg.data.down_ratio,
                self.cfg.data.input_h, self.cfg.data.input_w, self.cfg.data.scale_range,
                self.cfg.data.scale, self.cfg.test.test_rescale, self.cfg.data.test_scale,
                flip_type=self.cfg.data.flip_type, list_curved_py=np.array(list_curved_py),
                img_gt=gt_img, is_shift=self.cfg.data.augment_shift, gt_mask_labels=self.cfg.data.gt_mask_label,
                rot_type=self.cfg.data.augment_rotate, rot_angle=fixed_rot_angle
            )
        instance_polys = self.transform_original_data(instance_polys, flipped, height, width, trans_output, inp_out_hw, rot_angle=rot_angle)
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        # for deep snake
        if 'snake' in self.cfg.commen.task:
            extreme_points = self.get_extreme_points(instance_polys)

        #detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([len(self.json_category_id_to_continuous_id), output_h, output_w], dtype=np.float32)
        ct_cls = []
        wh = []
        ct_ind = []

        #segmentation
        img_gt_polys = []
        can_gt_polys = []
        keyPointsMask = []
        img_gt_init_polys = []
        can_gt_init_polys = []
        img_gt_coarse_polys = []
        can_gt_coarse_polys = []
        if 'snake' in self.cfg.commen.task:
            img_it_polys = []
            can_it_polys = []
            img_it_init_polys = []
            can_it_init_polys = []

        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                poly = make_simply_connected(poly)
                if isinstance(poly, (np.ndarray, list)):
                    is_poly_empty = len(poly) == 0
                else:
                    is_poly_empty = poly.is_empty
                if not is_poly_empty:
                    if isinstance(poly, list):
                        polys = poly
                    else:
                        polys = [poly]

                    if 'snake' in self.cfg.commen.task:
                        # print(len(polys))
                        extreme_points = [get_extreme_points(py) for py in polys]
                        # extreme_points = self.get_extreme_points(polys)
                    for k in range(len(polys)):
                        polyi = polys[k]
                        if 'snake' in self.cfg.commen.task:
                            extreme_point = extreme_points[k]
                        x_min, y_min = np.min(polyi[:, 0]), np.min(polyi[:, 1])
                        x_max, y_max = np.max(polyi[:, 0]), np.max(polyi[:, 1])
                        bbox = [x_min, y_min, x_max, y_max]
                        h, w = y_max - y_min + 1, x_max - x_min + 1
                        if h <= 1 or w <= 1:
                            continue
                        if (h*w < self.cfg.data.valid_box_area) and (x_min < self.cfg.data.valid_box_margin or y_min < self.cfg.data.valid_box_margin or abs(x_max - output_w) < self.cfg.data.valid_box_margin or abs(y_max - output_h) < self.cfg.data.valid_box_margin):
                            continue
                        decode_box = self.prepare_detection(bbox, polyi, ct_hm, cls_id, wh, ct_cls, ct_ind)
                        if 'snake' in self.cfg.commen.task:
                            self.prepare_init(decode_box, extreme_point, img_it_init_polys, can_it_init_polys, img_gt_init_polys, can_gt_init_polys)
                            self.prepare_snake_evolution(polyi, extreme_point, img_it_polys, can_it_polys, img_gt_polys, can_gt_polys)
                            img_gt_coarse_polys = img_gt_polys
                            can_gt_coarse_polys = can_gt_polys
                        else:
                            self.prepare_evolution(polyi, img_gt_polys, can_gt_polys, keyPointsMask, img_gt_init_polys, can_gt_init_polys, img_gt_coarse_polys, can_gt_coarse_polys)
                        # zoom_img, zoom_polys, init_ind_box, label_array, label_index_array = self.prepare_polyrnn(img, poly)

        data_input.update({'inp': inp, 'orig_img': orig_img})
        if inp_gt is not None:
            data_input.update({'pixel_gt': inp_gt})
        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        evolution = {'img_gt_polys': img_gt_polys, 'can_gt_polys': can_gt_polys,
                     'img_gt_init_polys': img_gt_init_polys, 'can_gt_init_polys': can_gt_init_polys,
                     'img_gt_coarse_polys': img_gt_coarse_polys, 'can_gt_coarse_polys': can_gt_coarse_polys}
        if 'snake' in self.cfg.commen.task:
            evolution.update({'img_it_init_polys': img_it_init_polys, 'can_it_init_polys': can_it_init_polys, 'img_it_polys': img_it_polys, 'can_it_polys': can_it_polys})
        # pyrnn = {'poly_label_array': label_array, 'poly_index_target': label_index_array}
        data_input.update(detection)
        data_input.update(evolution)
        if self.cfg.data.get_keypoints_mask and ('snake' not in self.cfg.commen.task):
            data_input.update({'keypoints_mask': keyPointsMask})
        # data_input.update(pyrnn)
        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': image_id, 'ann': ann, 'ct_num': ct_num}
        data_input.update({'meta': meta})
        if 'snake' in self.cfg.commen.task:
            data_input.update({'num_points_init': self.cfg.commen.init_points_per_poly,
                               'num_points_coarse': self.cfg.data.points_per_poly,
                               'num_points': self.cfg.data.points_per_poly})
        else:
            data_input.update({'num_points_init': int(self.cfg.model.heads['wh']/2) if 'wh' in self.cfg.model.heads else self.cfg.commen.init_points_per_poly, 'num_points_coarse': self.cfg.commen.init_points_per_poly, 'num_points': self.cfg.data.points_per_poly})
        return data_input

    def __len__(self):
        if self.cfg.data.augment_rotate == 'all_4':
            return len(self.anns) * 4
        else:
            return len(self.anns)


class RasterDataset(data.Dataset):
    def __init__(self, anno_file_list, root_dir, split, cfg):
        super(RasterDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.anno_file_list = anno_file_list
        self.root_dir = root_dir

        self.pys = self._get_datset()

    def _get_datset(self):
        pys = []
        for ann_file in self.anno_file_list:
            with open(os.path.join(self.root_dir, f'{ann_file}.pickle'), 'rb') as fr:
                pys_dict = pickle.load(fr)
            pys.extend(pys_dict[self.split])
        return pys


    def transform_original_data(self, instance_polys, flipped, height, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]
            if (isinstance(flipped, bool) and flipped) or flipped == 'lr':
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_
            elif flipped == 'ud':
                polys_ = []
                for poly in polys:
                    poly[:, 1] = height - np.array(poly[:, 1]) - 1
                    polys_.append(poly.copy())
                polys = polys_
            elif flipped == 'udlr':
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    poly[:, 1] = height - np.array(poly[:, 1]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = filter_tiny_polys(instance)
            polys = get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def __getitem__(self, index):
        data_input = {}
        meta = {'img_name': index}
        if self.split in ('test','val'):
            meta.update({'test': ''})
        data_input.update({'meta': meta})
        data_input.update({'pys': self.pys[index], 'ct_img_idx': index})
        return data_input

    def __len__(self):
        return len(self.pys)