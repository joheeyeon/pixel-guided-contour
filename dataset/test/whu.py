# from .base import Dataset
import os
import numpy as np
from pycocotools.coco import COCO
from ..train.utils import transform_polys, filter_tiny_polys, get_cw_polys, gaussian_radius, draw_umich_gaussian,\
uniformsample, four_idx, get_img_gt, img_poly_to_can_poly, augment
from ..train.whu import WhuDataset

# class SbdTestDataset(Dataset):
#     pass

class WhuTestDataset(WhuDataset):
    def process_info(self, ann):
        image_id = ann
        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=0)
        image_name = self.coco.loadImgs(int(image_id))[0]['file_name']
        image_path = os.path.join(self.data_root, self.coco.loadImgs(int(image_id))[0]['file_name'])
        ann = self.coco.loadAnns(ann_ids)
        return ann, image_path, image_id, image_name

    def __getitem__(self, index):
        data_input = {}

        ann = self.anns[index]
        anno, image_path, image_id, image_name = self.process_info(ann)
        img, instance_polys, cls_ids = self.read_original_data(anno, image_path)
        if ('pixel' in self.cfg.model.heads) or (self.cfg.model.type_add_pixel_mask == 'concat'):
            img_gt = self.get_gt_img(ann)
        else:
            img_gt = None
        width, height = img.shape[1], img.shape[0]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw, inp_gt, rot_angle = \
            augment(
                img, self.split,
                self.cfg.data.data_rng, self.cfg.data.eig_val, self.cfg.data.eig_vec,
                self.cfg.data.mean, self.cfg.data.std, self.cfg.data.down_ratio,
                self.cfg.data.input_h, self.cfg.data.input_w, self.cfg.data.scale_range,
                self.cfg.data.scale, self.cfg.test.test_rescale, self.cfg.data.test_scale,
                flip_type=self.cfg.data.flip_type,
                img_gt=img_gt, is_shift=self.cfg.data.augment_shift, gt_mask_labels=self.cfg.data.gt_mask_label
            )
        # os.makedirs("data/check/inp_gt/whu", exist_ok=True)
        # import cv2
        # cv2.imwrite(f"data/check/inp_gt/whu/{image_name.split('.')[0]}.png", inp_gt*255)
        instance_polys = self.transform_original_data(instance_polys, flipped, height, width, trans_output, inp_out_hw)
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
                if 'snake' in self.cfg.commen.task:
                    extreme_point = extreme_points[i][j]
                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue
                decode_box = self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                if 'snake' in self.cfg.commen.task:
                    self.prepare_init(decode_box, extreme_point, img_it_init_polys, can_it_init_polys,
                                      img_gt_init_polys, can_gt_init_polys)
                    self.prepare_snake_evolution(poly, extreme_point, img_it_polys, can_it_polys, img_gt_polys,
                                                 can_gt_polys)
                    img_gt_coarse_polys = img_gt_polys
                    can_gt_coarse_polys = can_gt_polys
                else:
                    self.prepare_evolution(poly, img_gt_polys, can_gt_polys, keyPointsMask, img_gt_init_polys, can_gt_init_polys, img_gt_coarse_polys, can_gt_coarse_polys)

        data_input.update({'inp': inp})
        if inp_gt is not None:
            data_input.update({'pixel_gt': inp_gt})
        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        evolution = {'img_gt_polys': img_gt_polys, 'can_gt_polys': can_gt_polys,
                     'img_gt_init_polys': img_gt_init_polys, 'can_gt_init_polys': can_gt_init_polys,
                     'img_gt_coarse_polys': img_gt_coarse_polys, 'can_gt_coarse_polys': can_gt_coarse_polys}
        if 'snake' in self.cfg.commen.task:
            evolution.update({'img_it_init_polys': img_it_init_polys, 'can_it_init_polys': can_it_init_polys, 'img_it_polys': img_it_polys, 'can_it_polys': can_it_polys})
        data_input.update(detection)
        data_input.update(evolution)
        if self.cfg.data.get_keypoints_mask:
            data_input.update({'keypoints_mask': keyPointsMask})
        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': image_id, 'ann': ann, 'ct_num': ct_num, 'test': '', 'img_name': image_name}
        data_input.update({'meta': meta})
        if 'snake' in self.cfg.commen.task:
            data_input.update({'num_points_init': self.cfg.commen.init_points_per_poly,
                               'num_points_coarse': self.cfg.data.points_per_poly,
                               'num_points': self.cfg.data.points_per_poly})
        else:
            data_input.update({'num_points_init': int(self.cfg.model.heads['wh']/2) if 'wh' in self.cfg.model.heads else self.cfg.commen.init_points_per_poly, 'num_points_coarse': self.cfg.commen.init_points_per_poly, 'num_points': self.cfg.data.points_per_poly})
        return data_input

