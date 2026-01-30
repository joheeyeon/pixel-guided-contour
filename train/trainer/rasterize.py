import torch.nn as nn
from .utils import FocalLoss, sigmoid, DMLoss, rasterize_instances, MultiFocalLoss
import torch
# from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask
import cv2, random
import numpy as np
from polygenerator import random_polygon

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

class NetworkWrapper(nn.Module):
    def __init__(self, net, cfg=None):
        super(NetworkWrapper, self).__init__()
        self.net = net
        # self.with_dml = with_dml
        # self.dml_start_epoch = dml_start_epoch
        # self.weight_dict = weight_dict
        self.cfg = cfg
        self.eps = 1e-8
        if 'reg' in self.cfg.model.raster_netparams:
            self.is_reg = self.cfg.model.raster_netparams['reg']
        else:
            self.is_reg = True

        if self.cfg.train.loss_type['raster'] in (None, 'regfocal'):
            self.map_crit = FocalLoss()
        elif self.cfg.train.loss_type['raster'] in ('crossentropy',):
            self.map_crit = nn.CrossEntropyLoss()
        else: #'clsfocal'
            self.map_crit = MultiFocalLoss(alpha=0.5)

        # if self.clip_to_proposal or not self.use_rasterized_gt:
        #     self.clipper = ClippingStrategy(cfg)
        #     self.gt_rasterizer = None
        # else:
        #     self.gt_rasterizer = SoftPolygon(inv_smoothness=1.0, mode="hard_mask")

    @torch.no_grad()
    def _create_targets(self, instances, img_hw, lid=0):
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
            mask = np.zeros((img_hw[0], img_hw[1], 1), np.uint8)
            masks.append(cv2.fillPoly(mask, [instance], 1))
        masks = np.stack(masks, axis=0) #(Nc, H, W, 1)
        if self.is_reg:
            return masks.transpose((0, 3, 1, 2))
        else:
            return masks.squeeze(-1)
        # return masks.unsqueeze(1).float()

    @torch.no_grad()
    def _generate_random_contour(self, n_contour, range_n_init_vertex, n_out_vertex):
        img_size = [int(self.cfg.data.input_h/self.cfg.data.down_ratio),
                    int(self.cfg.data.input_w/self.cfg.data.down_ratio)] if self.cfg.model.is_raster_down_sampled \
            else [self.cfg.data.input_h, self.cfg.data.input_w]
        contours = []
        for ci in range(n_contour):
            n_init_vertex = random.randint(range_n_init_vertex[0], range_n_init_vertex[-1])
            polygon = random_polygon(num_points=n_init_vertex)
            # scale to * (random)size + (random)center
            py_h = random.randint(5, img_size[0])
            py_w = random.randint(5, img_size[1])
            st_x =  random.randint(0, img_size[1] - py_w)
            st_y = random.randint(0, img_size[0] - py_h)
            polygon = np.array(polygon) * np.array([py_w, py_h]) + np.array([st_x, st_y])
            # upsample to n_out_vertex
            polygon = uniformsample(polygon, n_out_vertex)
            if 'apply_small_movings' in self.cfg.data.contour_param:
                if self.cfg.data.contour_param['apply_small_movings']:
                    max_var = np.random.randint(0, 5 if not 'max_var_upper' in self.cfg.data.contour_param else self.cfg.data.contour_param['max_var_upper'])
                    if max_var > 0:
                        small_movings = np.random.randint(low=-max_var, high=max_var, size=polygon.shape)
                        polygon += small_movings

            contours.append(polygon)
        return np.stack(contours, axis=0)

    # @torch.no_grad()
    # def get_bitmasks(self, instances, img_h, img_w):
    #     gt_masks = []
    #     for per_im_gt_inst in instances:
    #         if not per_im_gt_inst.has("gt_masks"):
    #             continue
    #         # start = int(self.mask_stride // 2)
    #         start = int(self.mask_stride_sup // 2)
    #         if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
    #             polygons = per_im_gt_inst.get("gt_masks").polygons
    #             per_im_bitmasks = []
    #             # per_im_bitmasks_full = []
    #             for per_polygons in polygons:
    #                 bitmask = polygons_to_bitmask(per_polygons, img_h, img_w)
    #                 # TODO: 这里可以直接转换低分辨率的mask? 这里好像没有对齐？应该先到原图大小，然后再padding到原图大小？
    #                 bitmask = torch.from_numpy(bitmask).to(self.device).float()
    #                 bitmask = bitmask[start::self.mask_stride_sup, start::self.mask_stride_sup]
    #                 assert bitmask.size(0) * self.mask_stride_sup == img_h
    #                 assert bitmask.size(1) * self.mask_stride_sup == img_w
    #                 per_im_bitmasks.append(bitmask)
    #
    #             gt_masks.append(torch.stack(per_im_bitmasks, dim=0))
    #         else:  # RLE format bitmask
    #             bitmasks = per_im_gt_inst.get("gt_masks").tensor
    #             h, w = bitmasks.size()[1:]
    #             # pad to new size
    #             bitmasks_full = F.pad(bitmasks, (0, img_w - w, 0, img_h - h), "constant", 0)
    #             bitmasks = bitmasks_full[:, start::self.mask_stride_sup, start::self.mask_stride_sup]
    #
    #             gt_masks.append(bitmasks)
    #     return torch.cat(gt_masks, dim=0)  # (N, H, W)

    def forward(self, batch):
        if 'img_gt_polys' in batch:
            in_pys = batch['img_gt_polys'][batch['ct_01']]
            if not self.cfg.model.is_raster_down_sampled:
                in_pys *= self.cfg.commen.down_ratio
        else:
            in_pys = batch['pys']
            if self.cfg.model.is_raster_down_sampled:
                in_pys /= self.cfg.commen.down_ratio

        if 'test' in batch['meta']:
            output = self.net(in_pys)
            return output
        else:
            if self.cfg.train.raster_add_random_contour and (in_pys.size(0) < self.cfg.train.contour_batch_size):
                n_contour_add = self.cfg.train.contour_batch_size - in_pys.size(0)
                py_add = self._generate_random_contour(n_contour_add, self.cfg.train.raster_add_range_init_vertex, self.cfg.commen.points_per_poly) #n_py x n_vert x 2
                in_pys = torch.cat([in_pys, torch.from_numpy(py_add).to(device=in_pys.device, dtype=in_pys.dtype)], dim=0)

            output = self.net(in_pys)
            epoch = batch['epoch']
            scalar_stats = {}
            loss = 0.

            # rasterize
            # get mask
            raster_map_size = [int(self.cfg.data.input_w/self.cfg.commen.down_ratio), int(self.cfg.data.input_h/self.cfg.commen.down_ratio)] if self.cfg.model.is_raster_down_sampled \
                else [self.cfg.data.input_w, self.cfg.data.input_h]
            gt_masks = self._create_targets(in_pys.clone().detach().cpu().numpy(),
                                            raster_map_size)
            if self.is_reg:
                pred_mask = nn.functional.sigmoid(output['mask']) + self.eps
                gt_mask_tensor = torch.from_numpy(gt_masks).to(output['mask'].device).float()
            else:
                gt_mask_tensor = torch.from_numpy(gt_masks).to(output['mask'].device).long()
                pred_mask = output['mask']

            map_loss = self.map_crit(pred_mask, gt_mask_tensor)
            loss += map_loss
            scalar_stats.update({'map_loss': map_loss})

            # for poly-lstm
            # py_loss = 0
            # n = len(output['py_pred']) - 1 if self.with_dml else len(output['py_pred'])
            # for i in range(n):
            #     if num_polys == 0:
            #         part_py_loss = torch.sum(output['py_pred'][i]) * 0.0
            #     else:
            #         part_py_loss = self.py_crit(output['py_pred'][i], output['img_gt_polys'])
            #     py_loss += part_py_loss / len(output['py_pred'])
            #     scalar_stats.update({'py_loss{}'.format(i): part_py_loss})
            # loss += py_loss * self.weight_dict['evolve']
            #
            # if self.with_dml:
            #     if epoch >= self.dml_start_epoch:
            #         dm_loss = self.dml_crit(output['py_pred'][-2],
            #                                 output['py_pred'][-1],
            #                                 output['img_gt_polys'],
            #                                 keyPointsMask)
            #         scalar_stats.update({'py_final_loss': dm_loss})
            #         loss += dm_loss / len(output['py_pred']) * self.weight_dict['evolve']
            #     else:
            #         py_last_loss = self.py_crit(output['py_pred'][-1], output['img_gt_polys'])
            #         scalar_stats.update({'py_final_loss': py_last_loss})
            #         loss += py_last_loss / len(output['py_pred']) * self.weight_dict['evolve']

            scalar_stats.update({'loss': loss})

            return output, loss, scalar_stats

