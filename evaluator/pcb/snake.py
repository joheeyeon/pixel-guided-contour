import os, glob
import json
from . import utils
import pycocotools.coco as coco
from .cocoeval import COCOeval
import torch
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch.distributed as dist

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

@rank_zero_only
def save_json(data, path):
    import json
    with open(path, 'w') as f:
        json.dump(data, f)

def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0

class Evaluator:
    def __init__(self, result_dir, anno_file, cfg):
        self.results = []
        self.results_init = []
        self.img_ids = []
        self.aps = []
        self.calc_deform_metric = cfg.calc_deform_metric if hasattr(cfg, 'calc_deform_metric') else False
        self.mask_deforms = {}
        self.cfg = cfg

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        ann_file = anno_file
        self.coco = coco.COCO(ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.max_eps = cfg.max_eps if hasattr(cfg, 'max_eps') else 1.
        self.th_iou = cfg.th_iou if hasattr(cfg, 'th_iou') else 0.99

    def reset(self):
        self.results = []
        self.img_ids = []

    def evaluate(self, output, batch):
        # ✅ 배치 처리를 위해 모든 인스턴스의 예측 결과를 가져옵니다.
        all_detections = output['detection']
        all_polys = output['py'][-1]
        batch_indices = output.get('batch_ind')

        # 배치 내에 감지된 인스턴스가 없으면 종료합니다.
        if all_detections.shape[0] == 0:
            return

        batch_size = batch['inp'].size(0)
        h, w = batch['inp'].size(-2), batch['inp'].size(-1)

        # ✅ 배치 내의 각 이미지를 순회하며 처리합니다.
        for b_idx in range(batch_size):
            # 현재 이미지(b_idx)에 해당하는 인스턴스의 인덱스를 찾습니다.
            instance_indices = (batch_indices == b_idx).nonzero(as_tuple=True)[0]

            # 현재 이미지에 감지된 인스턴스가 없으면 다음 이미지로 넘어갑니다.
            if len(instance_indices) == 0:
                continue

            # 현재 이미지에 해당하는 데이터만 필터링합니다.
            detections_for_img = all_detections[instance_indices]
            polys_for_img = all_polys[instance_indices]

            if 'snake' in self.cfg.task:
                score = detections_for_img[:, 4].detach().cpu().numpy()
                label = detections_for_img[:, 5].detach().cpu().numpy().astype(int)
                py = polys_for_img.detach().cpu().numpy() * self.cfg.down_ratio
            elif 'maskinit' in self.cfg.task:
                score = detections_for_img[:, 0].detach().cpu().numpy()
                label = detections_for_img[:, 1].detach().cpu().numpy().astype(int)
                py = polys_for_img.detach().cpu().numpy()
            else:
                score = detections_for_img[:, 2].detach().cpu().numpy()
                label = detections_for_img[:, 3].detach().cpu().numpy().astype(int)
                py = polys_for_img.detach().cpu().numpy()

            # 현재 이미지의 메타데이터를 안전하게 추출합니다.
            img_id = int(batch['meta']['img_id'][b_idx].item())
            center = batch['meta']['center'][b_idx].detach().cpu().numpy()
            scale = batch['meta']['scale'][b_idx].detach().cpu().numpy()

            trans_output_inv = utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
            img = self.coco.loadImgs(img_id)[0] if isinstance(self.coco.loadImgs(img_id), list) else self.coco.loadImgs(img_id)
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
            if img_id not in self.img_ids:
                self.img_ids.append(img_id)

            # for deform
            # if self.calc_deform_metric:
            #     polys_init_for_img = output['poly_coarse'][instance_indices]
            #     py_init = polys_init_for_img.detach().cpu().numpy()
            #     py_init = [utils.affine_transform(py_, trans_output_inv) for py_ in py_init]
            #     rles_init = utils.coco_poly_to_rle(py_init, ori_h, ori_w)
            #     coco_dets_init = []
            #     for i in range(len(rles_init)):
            #         detection_init = {
            #             'image_id': img_id,
            #             'category_id': self.contiguous_category_id_to_json_id[label[i]],
            #             'segmentation': rles_init[i],
            #             'score': float('{:.2f}'.format(score[i]))
            #         }
            #         coco_dets_init.append(detection_init)
            #
            #     self.results_init.extend(coco_dets_init)

    def summarize(self, print_out=True):
        rank = dist.get_rank() if dist.is_initialized() else 0

        # rank별로 partial 파일 저장
        part_file = os.path.join(self.result_dir, f'results_rank{rank}.json')
        with open(part_file, 'w') as f:
            json.dump(self.results, f)

        # barrier sync
        if dist.is_initialized():
            dist.barrier()

        # rank0만 merge + 평가
        summ_dict = {}
        if rank == 0:
            all_results = []
            for fpath in glob.glob(os.path.join(self.result_dir, 'results_rank*.json')):
                with open(fpath, 'r') as f:
                    all_results.extend(json.load(f))

            merged_file = os.path.join(self.result_dir, 'results.json')
            with open(merged_file, 'w') as f:
                json.dump(all_results, f)

            # Check if there are any detection results
            if len(all_results) == 0:
                print("[WARNING] No detections found in Stage 2 testing. Skipping COCO evaluation.")
                # Return default values for all metrics
                for i, key in enumerate(AP_keys):
                    summ_dict[key] = 0.0
                summ_dict[f'mean IoU at{self.cfg.vis_th_score}(0)'] = 0.0
            else:
                coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
                coco_eval = COCOeval(self.coco, coco_dets, 'segm')
                coco_eval.params.imgIds = self.img_ids
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                iou = coco_eval.evaluateIoUAt(th_score=self.cfg.vis_th_score)
                self.aps.append(coco_eval.stats[0])
                for i in range(len(coco_eval.stats)):
                    summ_dict[AP_keys[i]] = coco_eval.stats[i]
                if iou is not None:
                    summ_dict[f'mean IoU at{self.cfg.vis_th_score}({iou.size})'] = iou.mean() if isinstance(iou, np.ndarray) else iou
            
            self.results = []
            self.img_ids = []

            # for deform
            # if self.calc_deform_metric:
            #     json.dump(self.results_init, open(os.path.join(self.result_dir, 'results_init.json'), 'w'))
            #     coco_dets_init = self.coco.loadRes(os.path.join(self.result_dir, 'results_init.json'))
            #     coco_eval_init = COCOeval(self.coco, coco_dets_init, 'segm')
            #     coco_eval_init.evaluate()
            #     ious_init = list(coco_eval_init.ious.values())
            #     ious_init = [j for sub in ious_init for j in sub]
            #     ious_init = [j for sub in ious_init for j in sub]
            #     ious_init = np.array(ious_init)
            #     ind_to_calc = np.nonzero((ious_init < self.th_iou) & (ious_init > 0))
            #     ious_final = list(coco_eval.ious.values())
            #     ious_final = [j for sub in ious_final for j in sub]
            #     ious_final = [j for sub in ious_final for j in sub]
            #     ious_final = np.array(ious_final)
            #     ious_diff = ious_final[ind_to_calc] - ious_init[ind_to_calc]
            #     mean_ious_diff = np.mean(ious_diff)
            #     return summ_dict, mean_ious_diff

        # 모든 rank에 summ_dict 전달
        if dist.is_initialized():
            obj_list = [summ_dict]
            dist.broadcast_object_list(obj_list, src=0)
            summ_dict = obj_list[0]

        return summ_dict, None

class DetectionEvaluator:
    def __init__(self, result_dir, anno_dir):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        ann_file = anno_dir
        self.coco = coco.COCO(ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def evaluate(self, output, batch):
        detection = output['detection']
        detection = detection[0] if detection.dim() == 3 else detection
        # box = detection[:, :4].detach().cpu().numpy() * snake_config.down_ratio
        score = detection[:, 2].detach().cpu().numpy()
        label = detection[:, 3].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach() if 'py' in output else output['ex'].detach()
        if len(py) == 0:
            return
        box = torch.cat([torch.min(py, dim=1, keepdim=True)[0], torch.max(py, dim=1, keepdim=True)[0]], dim=1)
        box = box.cpu().numpy()

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        if len(box) == 0:
            return

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']

        coco_dets = []
        for i in range(len(label)):
            box_ = utils.affine_transform(box[i].reshape(-1, 2), trans_output_inv).ravel()
            box_[2] -= box_[0]
            box_[3] -= box_[1]
            box_ = list(map(lambda x: float('{:.2f}'.format(x)), box_))
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'bbox': box_,
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def summarize(self, print_out=True):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'bbox')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        return {'ap': coco_eval.stats[0]}

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
