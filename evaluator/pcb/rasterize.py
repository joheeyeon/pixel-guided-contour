import os, cv2
import json
from . import utils
import pycocotools.coco as coco
from .cocoeval import COCOeval
import torch
import numpy as np
import torch.nn.functional as F

AP_keys = [
    "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
    "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
    "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
    "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
    "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
    "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"]

class Evaluator:
    def __init__(self, result_dir, cfg):
        self.results = []
        self.img_ids = []
        self.epoch = 0
        self.cfg = cfg
        # self.count_tmp = 0
        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

    def _create_targets(self, instances, img_wh):
        # if self.predict_in_box_space:
        #     if self.clip_to_proposal or not self.use_rasterized_gt:  # in coco, this is true
        #         clip_boxes = torch.cat([inst.proposal_boxes.tensor for inst in instances])
        #         masks = self.clipper(instances, clip_boxes=clip_boxes, lid=lid)  # bitmask
        #     else:
        #         masks = rasterize_instances(self.gt_rasterizer, instances, self.rasterize_at)
        # else:
        #     masks = self.get_bitmasks(instances, img_h=img_wh[1], img_w=img_wh[0])
        masks = []
        for obj_i in range(instances.shape[0]):
            instance = instances[obj_i].astype(np.int32)
            mask = np.zeros((img_wh[1], img_wh[0], 1), np.uint8)
            masks.append(cv2.fillPoly(mask, [instance], 1))
            # os.makedirs(f"{self.cfg.commen.result_dir}/CodeCheck/raster",exist_ok=True)
            # from scipy.io import savemat
            # savemat(f"{self.cfg.commen.result_dir}/CodeCheck/raster/{self.count_tmp}.mat",
            #         {'mask': cv2.fillPoly(mask, [instance], 1), 'py': instance})
            # self.count_tmp += 1

        if len(masks) > 0:
            masks = np.stack(masks, axis=0) #(Nc, H, W, 1)
            return masks.squeeze(-1)
        else:
            return np.zeros((0, img_wh[1], img_wh[0]), np.uint8)

    def evaluate(self, output, batch):
        if 'epoch' in batch:
            self.epoch = batch['epoch']
        else:
            self.epoch = 'val'
        img_id = batch['ct_img_idx']
        # pred = F.interpolate(output['mask'], scale_factor=self.cfg.data.down_ratio, mode='bilinear', align_corners=False).detach().cpu().numpy().squeeze().round()
        if 'reg' in self.cfg.model.raster_netparams:
            is_reg = self.cfg.model.raster_netparams['reg']
        else:
            is_reg = True
        if is_reg:
            pred = F.sigmoid(output['mask'].detach()).cpu().squeeze(1).numpy().round()
        else:
            pred = F.softmax(output['mask'].detach(), dim=1).cpu().numpy().round()[:,1,...]
        raster_out_size = [int(self.cfg.data.input_w / self.cfg.data.down_ratio),
                           int(self.cfg.data.input_h / self.cfg.data.down_ratio)] if self.cfg.model.is_raster_down_sampled else [
            self.cfg.data.input_w, self.cfg.data.input_h]
        if 'img_gt_polys' in batch:
            pys = batch['img_gt_polys'][batch['ct_01']]
        else:
            pys = batch['pys']
        label = self._create_targets(pys.clone().detach().cpu().numpy(), raster_out_size).astype(np.float32)
        # print(f"unique : pred = {np.unique(pred)} / label = {np.unique(label)}")
        # print(pred.shape, label.shape)
        tp = pred[label > 0].sum()
        fp = pred[label == 0].sum()
        fn = np.sum(label > 0) - tp
        tn = np.sum(label == 0) - fp
        # print(f"all : {pred.size} / tp : {tp} / fp : {fp} / fn : {fn} / tn : {tn}")

        stat = {'image_id': img_id, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
        self.results.append(stat)
        self.img_ids.append(img_id)

    def summarize(self):
        summ_dict = {'accuracy': 0., 'precision': 0., 'recall': 0., 'tp': 0., 'fp': 0., 'fn': 0., 'tn': 0.}
        for stat in self.results:
            summ_dict['tp'] += stat['tp']
            summ_dict['fp'] += stat['fp']
            summ_dict['fn'] += stat['fn']
            summ_dict['tn'] += stat['tn']

        summ_dict['precision'] = summ_dict['tp'] / (summ_dict['tp'] + summ_dict['fp'] + np.finfo(np.float32).eps)
        summ_dict['recall'] = summ_dict['tp'] / (summ_dict['tp'] + summ_dict['fn'] + np.finfo(np.float32).eps)
        summ_dict['f1-score'] = 2 * summ_dict['precision'] * summ_dict['recall'] / (summ_dict['precision'] + summ_dict['recall'] + np.finfo(np.float32).eps)
        summ_dict['accuracy'] = (summ_dict['tp']+summ_dict['tn']) / (summ_dict['tp'] + summ_dict['fp'] + summ_dict['tn'] + summ_dict['fn'])

        print(f"e{self.epoch}::")
        for summ_k, summ_v in summ_dict.items():
            print(f"[{summ_k}] {summ_v}")

        self.results = []
        self.img_ids = []
        return summ_dict, None


# class DeformEvaluator:
#     def __init__(self, result_dir, anno_file, max_eps=1., calc_deform_metric=True):
#         self.results = []
#         self.results_init = []
#         self.img_ids = []
#         self.aps = []
#         self.calc_deform_metric = calc_deform_metric
#         self.mask_deforms = {}
#
#         self.result_dir = result_dir
#         os.system('mkdir -p {}'.format(self.result_dir))
#
#         ann_file = anno_file
#         self.coco = coco.COCO(ann_file)
#
#         self.json_category_id_to_contiguous_id = {
#             v: i for i, v in enumerate(self.coco.getCatIds())
#         }
#         self.contiguous_category_id_to_json_id = {
#             v: k for k, v in self.json_category_id_to_contiguous_id.items()
#         }
#         self.max_eps = max_eps
#
#     def evaluate(self, output, batch):
#         detection = output['detection']
#         score = detection[:, 2].detach().cpu().numpy()
#         label = detection[:, 3].detach().cpu().numpy().astype(int)
#         py = output['py'][-1].detach().cpu().numpy()
#
#         if len(py) == 0:
#             return
#
#         img_id = int(batch['meta']['img_id'][0])
#         center = batch['meta']['center'][0].detach().cpu().numpy()
#         scale = batch['meta']['scale'][0].detach().cpu().numpy()
#
#         h, w = batch['inp'].size(2), batch['inp'].size(3)
#         trans_output_inv = utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
#         img = self.coco.loadImgs(img_id)[0]
#         ori_h, ori_w = img['height'], img['width']
#         py = [utils.affine_transform(py_, trans_output_inv) for py_ in py]
#         rles = utils.coco_poly_to_rle(py, ori_h, ori_w)
#
#         coco_dets = []
#         for i in range(len(rles)):
#             detection = {
#                 'image_id': img_id,
#                 'category_id': self.contiguous_category_id_to_json_id[label[i]],
#                 'segmentation': rles[i],
#                 'score': float('{:.2f}'.format(score[i]))
#             }
#             coco_dets.append(detection)
#
#         self.results.extend(coco_dets)
#         self.img_ids.append(img_id)
#         # for deform
#         if self.calc_deform_metric:
#             py_init = output['poly_coarse'].detach().cpu().numpy()
#             py_init = [utils.affine_transform(py_, trans_output_inv) for py_ in py_init]
#             rles_init = utils.coco_poly_to_rle(py_init, ori_h, ori_w)
#             coco_dets_init = []
#             for i in range(len(rles_init)):
#                 detection_init = {
#                     'image_id': img_id,
#                     'category_id': self.contiguous_category_id_to_json_id[label[i]],
#                     'segmentation': rles_init[i],
#                     'score': float('{:.2f}'.format(score[i]))
#                 }
#                 coco_dets_init.append(detection_init)
#
#             self.results_init.extend(coco_dets_init)
#
#     def summarize(self):
#         json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
#         coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
#         coco_eval = COCOeval(self.coco, coco_dets, 'segm')
#         coco_eval.params.imgIds = self.img_ids
#         coco_eval.evaluate()
#         # for deform
#         if self.calc_deform_metric:
#             json.dump(self.results_init, open(os.path.join(self.result_dir, 'results_init.json'), 'w'))
#             coco_dets_init = self.coco.loadRes(os.path.join(self.result_dir, 'results_init.json'))
#             coco_eval_init = COCOeval(self.coco, coco_dets_init, 'segm')
#             coco_eval_init.evaluate()
#             ious_init = np.array(coco_eval_init.ious.values())
#             ind_to_calc = np.where(ious_init < self.th_iou)
#             ious_final = np.array(coco_eval.ious.values())
#             ious_diff = ious_final[ind_to_calc]-ious_init[ind_to_calc]
#             mean_ious_diff = np.mean(ious_diff)
#
#
#         coco_eval.accumulate()
#         coco_eval.summarize()
#         self.results = []
#         self.img_ids = []
#         self.aps.append(coco_eval.stats[0])
#         summ_dict = {}
#         for i in range(len(coco_eval.stats)):
#             summ_dict[AP_keys[i]] = coco_eval.stats[i]
#         return summ_dict
