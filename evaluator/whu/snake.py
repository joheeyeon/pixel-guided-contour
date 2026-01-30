import os
import json, glob
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
        
        # Add 'info' key if missing (for WHU dataset compatibility with pycocotools)
        if 'info' not in self.coco.dataset:
            self.coco.dataset['info'] = {}

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.max_eps = cfg.max_eps if hasattr(cfg, 'max_eps') else 1.
        self.th_iou = cfg.th_iou if hasattr(cfg, 'th_iou') else 0.99
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        #========= edit:debug:sudden-kill:25-08-10 ============
        self.tmp_path = os.path.join(self.result_dir, f"val_results_rank{self.rank}.jsonl")
        open(self.tmp_path, "w").close()  # 초기화


    def reset(self):
        self.results = []
        self.img_ids = []
        # jsonl 파일 비우기
        open(self.tmp_path, "w").close()

    def evaluate(self, output, batch):
        # ✅ 배치 처리를 위해 모든 인스턴스의 예측 결과를 가져옵니다. (edit:debug:results-json-whu:25-08-10)
        all_detections = output['detection']
        all_polys = output['py'][-1]
        batch_indices = output.get('batch_ind')

        # 배치 내에 감지된 인스턴스가 없으면 종료합니다. (edit:debug:results-json-whu:25-08-10)
        if all_detections.shape[0] == 0:
            return

        batch_size = batch['inp'].size(0)
        h, w = batch['inp'].size(-2), batch['inp'].size(-1)

        # ✅ 배치 내의 각 이미지를 순회하며 처리합니다. (edit:debug:results-json-whu:25-08-10)
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
            img = self.coco.loadImgs(img_id)[0] if isinstance(self.coco.loadImgs(img_id), list) else self.coco.loadImgs(
                img_id)
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

            # self.results.extend(coco_dets)
            # evaluate() 끝부분에서 메모리에 쌓지 말고 바로 append
            with open(self.tmp_path, "a") as f:
                for det in coco_dets:
                    f.write(json.dumps(det) + "\n")
            if img_id not in self.img_ids:
                self.img_ids.append(img_id)

        # for deform
        # if self.calc_deform_metric:
        #     py_init = output['poly_coarse'].detach().cpu().numpy()
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
        #========= edit:debug:results-json-whu:25-08-10 ============
        # ========= edit:debug:sudden-kill:25-08-10 ============
        rank = dist.get_rank() if dist.is_initialized() else 0

        # 1) 각 rank가 자신의 img_ids를 파일로 저장 (중복 제거 후)
        ids_path = os.path.join(self.result_dir, f"img_ids_rank{rank}.txt")
        unique_ids = sorted(set(int(i) for i in self.img_ids))
        with open(ids_path, "w") as f:
            for i in unique_ids:
                f.write(f"{i}\n")

        # 2) 모두 쓰기 끝날 때까지 동기화
        if dist.is_initialized():
            dist.barrier()

        summ_dict = {}

        # 3) rank0만 merge + COCOeval 실행
        if rank == 0:
            # 3-1) JSONL 머지
            all_results = []
            for fpath in glob.glob(os.path.join(self.result_dir, "val_results_rank*.jsonl")):
                with open(fpath, "r") as f:
                    for line in f:
                        if line.strip():
                            all_results.append(json.loads(line))
            merged_file = os.path.join(self.result_dir, "results.json")
            with open(merged_file, "w") as f:
                json.dump(all_results, f)

            # 3-2) img_ids 머지
            all_img_ids = set()
            for fpath in glob.glob(os.path.join(self.result_dir, "img_ids_rank*.txt")):
                with open(fpath, "r") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            all_img_ids.add(int(s))
            all_img_ids = sorted(list(all_img_ids))

            # 3-3) COCOeval
            # print("[DEBUG]", len(all_results), len(all_img_ids))
            if len(all_results) > 0:
                coco_dets = self.coco.loadRes(merged_file)
                coco_eval = COCOeval(self.coco, coco_dets, 'segm')
                coco_eval.params.imgIds = all_img_ids
                coco_eval.evaluate()
                coco_eval.accumulate()
                if print_out:
                    coco_eval.summarize()

                # 필요 시 IoU 추가 계산 (너의 헬퍼 함수 유지)
                iou = coco_eval.evaluateIoUAt(th_score=self.cfg.vis_th_score)

                self.aps.append(coco_eval.stats[0])
                for i, k in enumerate(AP_keys):
                    summ_dict[k] = coco_eval.stats[i]
                if iou is not None:
                    summ_dict[f'mean IoU at{self.cfg.vis_th_score}({iou.size})'] = (
                        iou.mean() if isinstance(iou, np.ndarray) else iou
                    )
            else:
                # 검출이 전혀 없는 경우
                for i, k in enumerate(AP_keys):
                    summ_dict[k] = 0.0

            # (선택) 임시 파일 정리하고 싶으면 아래 주석 해제
            # for f in glob.glob(os.path.join(self.result_dir, "val_results_rank*.jsonl")): os.remove(f)
            # for f in glob.glob(os.path.join(self.result_dir, "img_ids_rank*.txt")): os.remove(f)

        # 4) rank0의 결과를 모두에게 전달
        if dist.is_initialized():
            obj_list = [summ_dict]
            dist.broadcast_object_list(obj_list, src=0)
            summ_dict = obj_list[0]

        # 내부 상태 초기화 (메모리 누수 방지)
        self.results = []
        self.img_ids = []

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
        
        # Add 'info' key if missing (for WHU dataset compatibility with pycocotools)
        if 'info' not in self.coco.dataset:
            self.coco.dataset['info'] = {}

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

    def summarize(self,print_out=True):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'bbox')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize(print_out=print_out)
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
