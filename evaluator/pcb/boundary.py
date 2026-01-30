import os
import json
from . import utils
from boundary_iou.coco_instance_api import coco
from boundary_iou.coco_instance_api.cocoeval import COCOeval
import torch

DILATION_RATIO = 0.05

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
    def __init__(self, result_dir, anno_file, cfg):
        self.results = []
        self.img_ids = []
        self.aps = []
        self.cfg = cfg

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        ann_file = anno_file
        self.coco = coco.COCO(ann_file, get_boundary=False, dilation_ratio=DILATION_RATIO)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def evaluate(self, output, batch):
        detection = output['detection']
        if 'snake' in self.cfg.task:
            score = detection[:, 4].detach().cpu().numpy()
            label = detection[:, 5].detach().cpu().numpy().astype(int)
            py = output['py'][-1].detach().cpu().numpy() * self.cfg.down_ratio if 'py' in output else output[
                                                                                                          'ex'].detach().cpu().numpy() * self.cfg.down_ratio
        elif 'maskinit' in self.cfg.task:
            score = detection[:, 0].detach().cpu().numpy()
            label = detection[:, 1].detach().cpu().numpy().astype(int)
            py = output['py'][-1].detach().cpu().numpy()
        else:
            score = detection[:, 2].detach().cpu().numpy()
            label = detection[:, 3].detach().cpu().numpy().astype(int)
            if isinstance(output['py'][-1], list):
                py = [py_.detach().cpu().numpy() for py_ in output['py'][-1]]
            else:
                py = output['py'][-1].detach().cpu().numpy()

        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        py = [utils.affine_transform(py_, trans_output_inv) for py_ in py]
        rles = utils.coco_poly_to_rle(py, ori_h, ori_w)

        coco_dets = []
        for i in range(len(rles)):
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'segmentation': rles[i],
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, iouType="boundary", dilation_ratio=DILATION_RATIO)
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        summ_dict = {}
        for i in range(len(coco_eval.stats)):
            summ_dict[AP_keys[i]] = coco_eval.stats[i]
        return summ_dict

