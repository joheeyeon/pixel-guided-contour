import os.path
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.init
import torch.utils.model_zoo as model_zoo
import wget
from torch import nn
import torchvision
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF
import cv2

from .backbone.dla import DLASeg
# from .data_utils import *
from .detector_decode.refine_decode import Decode
from .evolve.evolve_rnn import DecoderLSTM, RefineRNN
from .evolve.evolve_ccp import Evolution as EvolutionCCP, RefinePixel
from .evolve.evolve import Evolution
from .evolve.snake_evolve import Evolution as SnakeEvolution
from .detector_decode.utils import decode_ct_hm, clip_to_image, decode_ct_hm_snake
from .evolve.utils import collect_training
from .rasterize.rasterize import Rasterizer
from .data_utils import check_simply_connected, has_self_intersection
from .evolve.sharp import IPC
from .evolve.utils import get_normal_vec, prepare_training, global_to_local_ct_img_idx

DICT_dla_module = {'base': 'base', 'dlaup': 'dla_up', 'idaup': 'ida_up'}

def get_safe_num_groups(num_channels, max_groups=32):
    """
    num_channels을 나눌 수 있는 가장 큰 num_groups을 찾아 반환
    """
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return g
    return 1  # fallback

def convert_bn_to_gn(model, max_groups=32):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            # gn = nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)
            # setattr(model, name, gn)
            safe_groups = get_safe_num_groups(num_channels, max_groups)
            gn = nn.GroupNorm(num_groups=safe_groups, num_channels=num_channels)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(module, max_groups)

def rotate_batch_tensor(x, angle):
    # x: (B, C, H, W)
    rotated = []
    for img in x:
        rotated.append(TF.rotate(img, angle=angle, expand=False))
    return torch.stack(rotated, dim=0)

def inverse_rotate_contours(contours, angle, height, width):
    # contours: list of (V_i, 2) tensor
    rotated = []
    angle = angle % 360
    if angle == 0:
        return contours  # no rotation needed

    # 중심 좌표
    cx = (width - 1) / 2
    cy = (height - 1) / 2

    for contour in contours:
        xy = contour.clone().float()
        x = xy[:, 0] - cx
        y = xy[:, 1] - cy

        if angle == 270:
            x_new = y
            y_new = -x
        elif angle == 180:
            x_new = -x
            y_new = -y
        elif angle == 90:
            x_new = -y
            y_new = x
        else:
            raise ValueError("Only angles of 0, 90, 180, 270 are supported.")

        xy[:, 0] = x_new + cx
        xy[:, 1] = y_new + cy
        rotated.append(xy.round().long())
    return rotated


# def convert_bn_to_gn_in_dla(module, num_groups=4):
#     """
#     DLA 내부의 nn.BatchNorm2d 모듈만 GroupNorm(num_groups=4)로 교체.
#     inplace로 변경.
#
#     Args:
#         module (nn.Module): 전체 모델 또는 DLA 모듈
#         num_groups (int): GroupNorm의 그룹 수
#     """
#     for name, child in module.named_children():
#         if isinstance(child, nn.BatchNorm2d):
#             # num_channels 정보 필요
#             num_channels = child.num_features
#             gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
#
#             # 기존 모듈에 inplace 교체
#             setattr(module, name, gn)
#
#         else:
#             # recursive 적용
#             convert_bn_to_gn_in_dla(child, num_groups=num_groups)

class E2ECnet(nn.Module):
    def __init__(self, cfg=None):
        super(E2ECnet, self).__init__()
        self.cfg = cfg
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.use_gt_det = cfg.train.use_gt_det
        self.test_stage = cfg.test.test_stage

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          input_channels=4 if cfg.model.type_add_pixel_mask == 'concat' else 3,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          use_bn_in_head=cfg.model.use_bn_in_head,
                          interpolate_mode='nearest' if cfg.commen.deterministic_mode == "full" else 'bilinear',
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))

        c_in = self.dla.base.channels[self.dla.first_level]
        if self.cfg.model.concat_upper_layer is not None:
            c_in += getattr(self.dla, DICT_dla_module[self.cfg.model.concat_upper_layer.split('_')[0]]).channels[int(self.cfg.model.concat_upper_layer.split('_')[-1])]
        # 새로운 concat_multi_layers 채널 수 추가
        if getattr(self.cfg.model, 'concat_multi_layers', None) is not None:
            for layer_name in self.cfg.model.concat_multi_layers:
                _name, _idx = layer_name.split('_')[0], int(layer_name.split('_')[-1])
                c_in += getattr(self.dla, DICT_dla_module[_name]).channels[_idx]
        if self.cfg.model.cat_feature_with_pixelmap and self.cfg.model.cat_include_coarse:
            c_in += cfg.model.heads['pixel']
        if cfg.model.add_grad_feature:
            c_in += 1

        self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score, c_in=c_in,
                                    num_point_each_step=cfg.commen.points_per_poly_steps,
                                    with_img_idx=cfg.model.with_img_idx, refine_kernel_size=cfg.model.refine_kernel_size,
                                    use_dp=cfg.train.use_dp, cat_include_coarse=cfg.model.cat_include_coarse)
        self.gcn = Evolution(evole_ietr_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.commen.down_ratio, with_img_idx=cfg.model.with_img_idx, refine_kernel_size=cfg.model.refine_kernel_size,
                             in_featrue_dim=c_in, use_vertex_classifier=cfg.model.use_vertex_classifier)
        if self.cfg.model.with_rasterize_net:
            self.rasterizer = Rasterizer(rasterize_type=self.cfg.model.raster_type,
                                         sigma=self.cfg.model.raster_sigma,
                                         out_size=[int(self.cfg.data.input_w / self.cfg.commen.down_ratio),
                                                   int(self.cfg.data.input_h / self.cfg.commen.down_ratio)],
                                         scale=self.cfg.model.raster_scale,
                                         **self.cfg.model.raster_netparams)
        else:
            self.rasterizer = None

        if self.cfg.model.with_sharp_contour:
            if 'pixel' in cfg.model.heads:
                self.dim_reduce = nn.Sequential(
                    nn.Conv2d(self.dla.base.channels[self.dla.first_level] + cfg.model.heads[
                        'pixel'] if self.cfg.model.sharp_param['dim_reduce_after_cat'] else self.dla.base.channels[self.dla.first_level],
                              cfg.model.sharp_param['fine_dim'], kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True))
                self.ipc = IPC(input_dim=(cfg.model.sharp_param['fine_dim'] + 2) if self.cfg.model.sharp_param['dim_reduce_after_cat'] else (
                            cfg.model.sharp_param['fine_dim'] + 2 + cfg.model.heads['pixel']), num_params=cfg.model.sharp_param['ipc_num_params'],
                               ro=cfg.commen.down_ratio,
                               random_sample_range=cfg.model.sharp_param['ipc_random_sample_range'],
                               match_dist_p=cfg.model.sharp_param['ipc_match_dist_p'], dynamic=cfg.model.sharp_param['ipc_dynamic'])
            else:
                self.dim_reduce = nn.Sequential(
                    nn.Conv2d(self.dla.base.channels[self.dla.first_level], cfg.model.sharp_param['fine_dim'], kernel_size=3,
                              padding=1, bias=True),
                    nn.ReLU(inplace=True))
                self.ipc = IPC(input_dim=cfg.model.sharp_param['fine_dim'] + 2, num_params=cfg.model.sharp_param['ipc_num_params'],
                               ro=cfg.commen.down_ratio,
                               random_sample_range=cfg.model.sharp_param['ipc_random_sample_range'],
                               match_dist_p=cfg.model.sharp_param['ipc_match_dist_p'])

        self.net_preprocess()

    def forward(self, x, batch=None):
        if self.cfg.model.type_add_pixel_mask == 'concat':
            input = torch.cat([x,batch['pixel_gt'].to(x.device).unsqueeze(1)], dim=1)
        else:
            input = x
        output, cnn_feature, feature_banks = self.dla(input)
        output['feature_banks'] = feature_banks
        output['cnn_feature'] = cnn_feature
        if self.cfg.model.cat_feature_with_pixelmap:
            if self.cfg.model.cat_feature_normalized:
                pixel_map = torch.nn.functional.softmax(output['pixel'], dim=1)
            else:
                pixel_map = output['pixel']
            if pixel_map.shape[2:] == cnn_feature.shape[2:]:
                add_feature = pixel_map
            else:
                #edit to nearest (for reprod)
                # with torch.backends.cudnn.flags(enabled=True, deterministic=False):
                add_feature = torch.nn.functional.interpolate(pixel_map,
                                                              size=cnn_feature.shape[2:], mode='nearest')
            if self.cfg.model.cut_grad_add_feature:
                feature_deform = torch.cat((cnn_feature, add_feature.detach()), 1)
            else:
                feature_deform = torch.cat((cnn_feature, add_feature), 1)

        else:
            feature_deform = cnn_feature

        if self.cfg.model.cat_include_coarse:
            feature_deform_coarse = feature_deform
        else:
            feature_deform_coarse = cnn_feature

        if 'test' not in batch['meta']:
            feature_coarse, output = self.train_decoder(batch, feature_deform_coarse, output, is_training=True)
        else:
            with torch.no_grad():
                if self.test_stage == 'init':
                    ignore = True
                else:
                    ignore = False
                feature_coarse, output = self.train_decoder(batch, feature_deform_coarse, output, is_training=False, ignore_gloabal_deform=ignore,
                                                    get_feature=self.cfg.test.get_featuremap)
                if self.cfg.test.get_featuremap:
                    output['fm'] = {}
                    for k in output['feature_banks'].keys():
                        if isinstance(output['feature_banks'][k], list):
                            for banks_i in range(len(output['feature_banks'][k])):
                                output['fm'].update({f'F_{k}{banks_i}': output['feature_banks'][k][banks_i].clone().detach()})
                        else:
                            output['fm'].update({f'F_{k}': output['feature_banks'][k].clone().detach()})

                    output['fm'].update({'F_backbone': feature_deform.clone().detach()})
                    output['fm'].update({'F_coarse': feature_coarse.clone().detach()})

        output['feature_coarse'] = feature_coarse
        output = self.gcn(output, feature_deform, batch, test_stage=self.test_stage, cfg=self.cfg)

        if self.cfg.model.with_sharp_contour:
            if 'pixel' in output:
                if self.cfg.model.cat_feature_normalized:
                    pixel_map = torch.nn.functional.softmax(output['pixel'], dim=1)
                else:
                    pixel_map = output['pixel']
                if pixel_map.shape[2:] == cnn_feature.shape[2:]:
                    add_feature = pixel_map
                else:
                    # edit to nearest (for reprod)
                    # with torch.backends.cudnn.flags(enabled=True, deterministic=False):
                    add_feature = torch.nn.functional.interpolate(pixel_map,
                                                                  size=cnn_feature.shape[2:], mode='nearest')
                if self.cfg.model.sharp_param['dim_reduce_after_cat']:
                    feature_concat = torch.cat((cnn_feature, add_feature), 1)
                    fine_grained_feature = self.dim_reduce(feature_concat)
                else:
                    fine_grained_feature = self.dim_reduce(cnn_feature)
                    fine_grained_feature = torch.cat((fine_grained_feature, add_feature), 1)
            else:
                fine_grained_feature = self.dim_reduce(cnn_feature)

            if self.training:
                n_iter_ipc = self.cfg.train.sharp_param['n_iter_train_ipc_rancom_sample']
                for iter_random in range(n_iter_ipc):
                    output = self.ipc(output, fine_grained_feature, batch, train_gt=self.cfg.train.sharp_param['train_gt'], cut_py_grad=False, cut_feature_grad=False)
            else:
                pre_uncertainty = None
                mask_stop_refine = torch.zeros_like(output['py_pred' if 'py_pred' in output else 'py'][-1])
                for iter in range(self.cfg.model.sharp_param['refine_iters']):
                    output = self.ipc(output, fine_grained_feature, batch,
                                      train_gt=self.cfg.train.sharp_param['train_gt'] if self.training else False,
                                      cut_py_grad=False, cut_feature_grad=False)
                    output, pre_uncertainty, mask_stop_refine = self.sharp_refine(output, iter, pre_uncertainty,
                                                                                  mask_stop_refine,
                                                                                  constant=self.cfg.model.sharp_param['refine_constant'],
                                                                                  ipc_name='ipc_random' if self.cfg.train.sharp_param['train_gt'] and self.training else 'ipc',
                                                                                  batch=batch)

        if (not self.training) and self.cfg.test.check_simple and (output['py'][-1].shape[0] > 0):
            output['is_simple'] = check_simply_connected(output['py'][-1])

        if self.rasterizer is not None:
            output['pred_mask'] = {'init': self.rasterizer(output['poly_init']),
                                   'coarse': self.rasterizer(output['poly_coarse'])}
            if self.training:
                py_name = 'py_pred'
            else:
                py_name = 'py'
            for pi in range(len(output[py_name])):
                output['pred_mask'].update({f'py{pi}': self.rasterizer(output[py_name][pi])})
        return output

    def sharp_refine(self, output, iter, pre_uncertainty, mask_stop_refine, constant=0.03, max_step_num=5, ipc_name='ipc', batch=None):
        py_pred = output['py_pred' if 'py_pred' in output else 'py'][-1] / self.cfg.commen.down_ratio
        if self.cfg.model.sharp_param['py_refine_init'] == 'add_zero':
            py_refined = py_pred + 0.0
        else:
            py_refined = py_pred.clone()

        if self.cfg.train.sharp_param['fix_step_number']:
            step_num = self.cfg.train.sharp_param['step_num_init']
        else:
            step_num = max(iter+self.cfg.train.sharp_param['step_num_init'], max_step_num)
        area_box = (torch.max(py_pred[:,:,0],1)[0]-torch.min(py_pred[:,:,0],1)[0]) * (torch.max(py_pred[:,:,1],1)[0]-torch.min(py_pred[:,:,1],1)[0])
        if self.cfg.model.sharp_param['refine_normal_outward']:
            uncertainty = 0.5 - nn.functional.sigmoid(output[ipc_name][iter]).squeeze(1)
        else:
            uncertainty = nn.functional.sigmoid(output[ipc_name][iter]).squeeze(1) - 0.5

        if pre_uncertainty is not None:
            mask_do_refine = torch.logical_and(torch.logical_not(mask_stop_refine), torch.logical_and(torch.round(pre_uncertainty * uncertainty * 100) > 0,
                                                                                   torch.round(uncertainty*10) != 0).unsqueeze(-1).expand(-1,-1,2)) # revised:231020
        else:
            mask_do_refine = torch.logical_and(torch.logical_not(mask_stop_refine), (torch.round(uncertainty*10) != 0).unsqueeze(-1).expand(-1,-1,2))

        # constant = 0.1
        moving_step = (constant * torch.sqrt(area_box.unsqueeze(-1)) * uncertainty).unsqueeze(-1).expand(-1,-1,2)
        # sqrt_area = torch.sqrt(area_box.unsqueeze(-1).unsqueeze(-1).expand(-1,128,2))
        # print(f"sqrt(area) : {torch.sqrt(area_box)}")
        # print(f"moving_step : {moving_step[mask_do_refine]}")
        direction, norm_nxt, norm_pre = get_normal_vec(torch.roll(py_pred, 1, 1), py_pred, torch.roll(py_pred, -1, 1))
        # py_refined[mask_do_refine] = py_pred[mask_do_refine] + step_num * moving_step[mask_do_refine] * direction[mask_do_refine]
        py_refined[mask_do_refine] += step_num * moving_step[mask_do_refine] * direction[mask_do_refine]

        if 'mask_refine' in output:
            output['mask_refine'].append(mask_do_refine.clone())
        else:
            output.update({'mask_refine': [mask_do_refine.clone()]})

        if 'refine_uncertainty' in output:
            output['refine_uncertainty'].append(uncertainty.clone())
        else:
            output.update({'refine_uncertainty': [uncertainty.clone()]})

        output['py_pred' if 'py_pred' in output else 'py'].append(py_refined * self.cfg.commen.down_ratio)
        return output, uncertainty.clone(), torch.logical_not(mask_do_refine)

    def net_preprocess(self):
        fix_net_names = []
        if self.cfg.train.fix_dla:
            fix_net_names.append('dla')
        if self.cfg.train.fix_decode:
            fix_net_names.append('train_decoder')
        if self.cfg.train.fix_deform:
            fix_net_names.append('gcn')
        fix_network(self, fix_net_names)

class CCPnet(nn.Module):
    # combine contour and pixel net
    def __init__(self, cfg=None):
        super(CCPnet, self).__init__()
        self.cfg = cfg
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.use_gt_det = cfg.train.use_gt_det
        self.test_stage = cfg.test.test_stage
        self.use_pixel_on_init = getattr(cfg.model, 'use_pixel_on_init', False)
        self.pixel_concat_with_activation = getattr(cfg.model, 'pixel_concat_with_activation', False)

        if self.use_pixel_on_init:
            # use_pixel_on_init=True: DLASeg는 pixel head만 포함
            heads_dla = {'pixel': heads['pixel']}
        else:
            # 기존 방식: 모든 head 포함
            heads_dla = heads

        # DLA backbone 생성
        self.dla = DLASeg('dla{}'.format(num_layers), heads_dla,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          input_channels=4 if cfg.model.type_add_pixel_mask == 'concat' else 3,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          use_bn_in_head=cfg.model.use_bn_in_head,
                          interpolate_mode='nearest' if cfg.commen.deterministic_mode == "full" else 'bilinear',
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))
        
        # use_pixel_on_init가 True일 때 ct_hm, wh head를 별도로 생성
        if self.use_pixel_on_init:
            c_in = self.dla.base.channels[self.dla.first_level]
            if self.cfg.model.concat_upper_layer is not None:
                c_in += getattr(self.dla, DICT_dla_module[self.cfg.model.concat_upper_layer.split('_')[0]]).channels[
                    int(self.cfg.model.concat_upper_layer.split('_')[-1])]
            # 새로운 concat_multi_layers 채널 수 추가
            if getattr(self.cfg.model, 'concat_multi_layers', None) is not None:
                for layer_name in self.cfg.model.concat_multi_layers:
                    _name, _idx = layer_name.split('_')[0], int(layer_name.split('_')[-1])
                    c_in += getattr(self.dla, DICT_dla_module[_name]).channels[_idx]
            
            # pixel map과 concat될 feature 채널
            c_in_with_pixel = c_in + heads['pixel']
            
            # ct_hm, wh head를 위한 별도 레이어 (pixel과 concat 후 사용)
            self.init_heads = nn.ModuleDict()
            for head_name in ['ct_hm', 'wh']:
                classes = heads[head_name]
                fc = self._build_head_conv(c_in_with_pixel, classes, head_conv, head_name)
                self.init_heads[head_name] = fc

        # Decoder와 Evolution에 전달되는 feature는 실제로는 원본 backbone feature
        # 따라서 pixel concat은 Evolution 모듈 내부에서만 고려되어야 함
        c_in_base = self.dla.base.channels[self.dla.first_level]
        if self.cfg.model.concat_upper_layer is not None:
            c_in_base += getattr(self.dla, DICT_dla_module[self.cfg.model.concat_upper_layer.split('_')[0]]).channels[
                int(self.cfg.model.concat_upper_layer.split('_')[-1])]
        # 새로운 concat_multi_layers 채널 수 추가
        if getattr(self.cfg.model, 'concat_multi_layers', None) is not None:
            for layer_name in self.cfg.model.concat_multi_layers:
                _name, _idx = layer_name.split('_')[0], int(layer_name.split('_')[-1])
                c_in_base += getattr(self.dla, DICT_dla_module[_name]).channels[_idx]
        if cfg.model.add_grad_feature:
            c_in_base += 1

        # Decoder: 원본 feature 사용
        c_in_decoder = c_in_base
        if self.cfg.model.cat_feature_with_pixelmap and self.cfg.model.cat_include_coarse:
            c_in_decoder += cfg.model.heads['pixel']

        # Evolution: 내부에서 pixel concat이 수행되므로 base 채널만 전달
        # (Evolution 모듈 내부에서 channel_pixel을 더해 c_out_proj 계산)
        c_in_evolution = c_in_base

        self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score, c_in=c_in_decoder,
                                    num_point_each_step=cfg.commen.points_per_poly_steps,
                                    with_img_idx=cfg.model.with_img_idx, refine_kernel_size=cfg.model.refine_kernel_size,
                                    use_dp=cfg.train.use_dp, use_trans_feature=cfg.model.refine_use_trans_feature,
                                    cat_include_coarse=cfg.model.cat_include_coarse)
        self.gcn = EvolutionCCP(evolve_iter_num=getattr(cfg.model, 'evolve_iter_num', 1), evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.commen.down_ratio, with_img_idx=cfg.model.with_img_idx, refine_kernel_size=cfg.model.refine_kernel_size,
                             in_featrue_dim=c_in_evolution, use_vertex_classifier=cfg.model.use_vertex_classifier,
                                channel_pixel=1 if cfg.model.ccp_deform_pixel_norm=='argmax' else cfg.model.heads['pixel'],
                                c_out_proj=cfg.model.ccp_dim_out_proj, with_proj=cfg.model.ccp_with_proj, num_vertex=cfg.model.points_per_poly,
                                pixel_norm_type=cfg.model.ccp_deform_pixel_norm, vtx_cls_common_prediction_type=cfg.model.vtx_cls_common_prediction_type,
                                vtx_cls_kernel_size=cfg.model.vtx_cls_kernel_size, cfg=cfg,
                                gcn_weight_sharing=getattr(cfg.model, 'gcn_weight_sharing', True),
                                use_3x3_feature=getattr(cfg.model, 'use_3x3_feature', False),
                                feature_3x3_mode=getattr(cfg.model, 'feature_3x3_mode', 'flatten'))
        if cfg.model.use_refine_pixel:
            refine_dim_in = c_in_decoder
            if cfg.model.ccp_refine_pixel_input_norm == 'argmax':
                dim_pixel = 1
            else:
                dim_pixel = cfg.model.heads['pixel']
            refine_dim_in += (1+dim_pixel if cfg.model.ccp_refine_with_pre_pixel else 1)
            self.refine_pixel = RefinePixel(dim_out=cfg.model.heads['pixel'], down_ratio=cfg.commen.down_ratio,
                                            num_layers=cfg.model.refine_pixel_param['num_layers'] if 'num_layers' in cfg.model.refine_pixel_param else 2,
                                            dim_list=cfg.model.refine_pixel_param['dim_list'] if 'dim_list' in cfg.model.refine_pixel_param else [256],
                                            kernel_list=cfg.model.refine_pixel_param['kernel_list'] if 'kernel_list' in cfg.model.refine_pixel_param else [3,1],
                                            dim_in=refine_dim_in, input_norm_type=cfg.model.ccp_refine_pixel_input_norm,
                                            reduce_memory=cfg.model.contour_map_down_sample, refine_as_residual=cfg.model.ccp_refine_pixel_as_residual,
                                            convert_map_down_ratio=cfg.model.contour_map_down_ratio,
                                            module_structure=cfg.model.refine_pixel_param['module_structure'] if 'module_structure' in cfg.model.refine_pixel_param else None,
                                            contour_map_shape=[cfg.data.input_h//cfg.commen.down_ratio, cfg.data.input_w//cfg.commen.down_ratio],
                                            deterministic=cfg.commen.deterministic_mode)
        else:
            self.refine_pixel = None

        self.net_preprocess()

    def _build_head_conv(self, c_in, classes, head_conv, head_name):
        """head_conv 구조를 설정 가능하게 빌드"""
        # head_conv_config에서 설정 가져오기
        conv_config = getattr(self.cfg.model, 'head_conv_config', {})
        
        # 기본값 설정
        kernel_sizes = conv_config.get('kernel_sizes', [3, 1])
        channels = conv_config.get('channels', None)
        use_relu = conv_config.get('use_relu', [True])
        padding_mode = conv_config.get('padding', 'auto')
        
        # head별 개별 설정 지원
        if isinstance(conv_config, dict) and head_name in conv_config:
            head_specific = conv_config[head_name]
            kernel_sizes = head_specific.get('kernel_sizes', kernel_sizes)
            channels = head_specific.get('channels', channels)
            use_relu = head_specific.get('use_relu', use_relu)
            padding_mode = head_specific.get('padding', padding_mode)
        
        # channels 설정: None이면 기본값 [head_conv, classes] 사용
        if channels is None:
            if head_conv > 0:
                channels = [head_conv, classes]
            else:
                channels = [classes]
        else:
            channels = channels + [classes]  # 마지막에 classes 추가
        
        # use_relu 리스트 길이 조정 (마지막 layer는 항상 ReLU 없음)
        if len(use_relu) < len(kernel_sizes) - 1:
            use_relu = use_relu + [True] * (len(kernel_sizes) - 1 - len(use_relu))
        
        # 레이어 구성
        if head_conv > 0 or len(kernel_sizes) > 1:
            layers = []
            c_current = c_in
            
            for i, (kernel_size, c_out) in enumerate(zip(kernel_sizes, channels)):
                # padding 계산
                if padding_mode == 'auto':
                    padding = (kernel_size - 1) // 2
                else:
                    padding = padding_mode
                
                # Conv2d 추가
                layers.append(nn.Conv2d(c_current, c_out, kernel_size=kernel_size, 
                                       stride=1, padding=padding, bias=True))
                
                # ReLU 추가 (마지막 layer 제외)
                if i < len(channels) - 1 and i < len(use_relu) and use_relu[i]:
                    layers.append(nn.ReLU(inplace=True))
                
                c_current = c_out
            
            fc = nn.Sequential(*layers)
        else:
            # head_conv == 0인 경우: 단일 1x1 conv
            kernel_size = kernel_sizes[-1] if kernel_sizes else 1
            padding = (kernel_size - 1) // 2 if padding_mode == 'auto' else padding_mode
            fc = nn.Conv2d(c_in, classes, kernel_size=kernel_size, 
                          stride=1, padding=padding, bias=True)
        
        # hm head의 경우 bias 초기화
        if 'hm' in head_name:
            if isinstance(fc, nn.Sequential):
                fc[-1].bias.data.fill_(-2.19)
            else:
                fc.bias.data.fill_(-2.19)
        
        return fc

    def forward(self, x, batch=None):
        # print(f"[CCPNET] meta={batch.get('meta')}")
        if self.cfg.model.type_add_pixel_mask == 'concat':
            input = torch.cat([x,batch['pixel_gt'].to(x.device).unsqueeze(1)], dim=1)
        else:
            input = x
        if 'test' in batch['meta'] and self.cfg.test.single_rotate_angle is not None:
            input = torch.rot90(input, k=self.cfg.test.single_rotate_angle//90, dims=(2, 3))
        
        if self.use_pixel_on_init:
            # Step 1: backbone feature 추출 및 pixel head 실행
            output, cnn_feature, feature_banks = self.dla(input, save_banks=self.cfg.test.get_featuremap)
            # output에는 pixel만 있음
            
            # Step 2: pixel head output을 backbone feature와 concat
            pixel_map = output['pixel']
            if pixel_map.shape[2:] != cnn_feature.shape[2:]:
                pixel_map = torch.nn.functional.interpolate(
                    pixel_map,
                    size=cnn_feature.shape[2:],
                    mode='nearest'
                )
            
            # 옵션에 따라 pixel map에 activation 적용
            if self.pixel_concat_with_activation:
                pixel_map = torch.nn.functional.softmax(pixel_map, dim=1)
            
            # backbone feature와 pixel map concat
            feature_with_pixel = torch.cat([cnn_feature, pixel_map], dim=1)
            
            # Step 3: concat된 feature로 ct_hm, wh head 계산
            for head_name in ['ct_hm', 'wh']:
                output[head_name] = self.init_heads[head_name](feature_with_pixel)
            
            # cnn_feature는 원본 backbone feature 유지 (gcn 등에서 사용)
        else:
            # 기존 방식: 모든 head 동시 처리
            output, cnn_feature, feature_banks = self.dla(input, save_banks=self.cfg.test.get_featuremap)

        #not used recently
        if self.cfg.model.cat_feature_with_pixelmap:
            if self.cfg.model.cat_feature_normalized:
                pixel_map = torch.nn.functional.softmax(output['pixel'], dim=1)
            else:
                pixel_map = output['pixel']
            if pixel_map.shape[2:] == cnn_feature.shape[2:]:
                add_feature = pixel_map
            else:
                # edit to nearest (for reprod)
                # with torch.backends.cudnn.flags(enabled=True, deterministic=False):
                add_feature = torch.nn.functional.interpolate(pixel_map,
                                                              size=cnn_feature.shape[2:], mode='nearest')
            if self.cfg.model.cut_grad_add_feature:
                feature_deform = torch.cat((cnn_feature, add_feature.detach()), 1)
            else:
                feature_deform = torch.cat((cnn_feature, add_feature), 1)
        else:
            feature_deform = cnn_feature

        if self.cfg.model.cat_include_coarse:
            if self.cfg.model.ccp_deform_pixel_norm == 'softmax':
                pixel_map = F.softmax(output['pixel'], dim=1)
            elif self.cfg.model.ccp_deform_pixel_norm == 'trainable_softmax':
                # trainable_softmax의 경우 Evolution 모듈의 temperature 사용
                temperature = getattr(self.gcn, 'temperature', torch.ones(1, device=output['pixel'].device))
                pixel_map = F.softmax(output['pixel'] / temperature, dim=1)
            elif self.cfg.model.ccp_deform_pixel_norm == 'sep_trainable_sigmoid':
                # sep_trainable_sigmoid의 경우 Evolution 모듈의 sep_sigmoid 사용
                sep_sigmoid = getattr(self.gcn, 'sep_sigmoid', None)
                if sep_sigmoid is not None:
                    pixel_map = sep_sigmoid(output['pixel'])
                else:
                    pixel_map = torch.sigmoid(output['pixel'])  # fallback
            elif self.cfg.model.ccp_deform_pixel_norm == 'argmax':
                pixel_map = torch.argmax(output['pixel'], dim=1).unsqueeze(1)
            else:
                pixel_map = output['pixel']

            if pixel_map.shape[2:] == cnn_feature.shape[2:]:
                add_feature = pixel_map
            else:
                # edit to nearest (for reprod)
                # with torch.backends.cudnn.flags(enabled=True, deterministic=False):
                add_feature = torch.nn.functional.interpolate(pixel_map,
                                                              size=cnn_feature.shape[2:], mode='nearest')
            if self.cfg.model.cut_grad_add_feature:
                feature_deform_coarse = torch.cat((cnn_feature, add_feature.detach()), 1)
            else:
                feature_deform_coarse = torch.cat((cnn_feature, add_feature), 1)
        else:
            feature_deform_coarse = cnn_feature

        if 'test' not in batch['meta']:
            feature_coarse, output = self.train_decoder(batch, feature_deform_coarse, output, is_training=True)
        else:
            with torch.no_grad():
                if self.test_stage == 'init':
                    ignore = True
                else:
                    ignore = False
                feature_coarse, output = self.train_decoder(batch, feature_deform_coarse, output, is_training=False, ignore_gloabal_deform=ignore,
                                                    get_feature=self.cfg.test.get_featuremap)
                if (not self.training) and self.cfg.test.get_featuremap:
                    output['feature_banks'] = feature_banks
                    output['cnn_feature'] = cnn_feature
                    output['fm'] = {}
                    for k in output['feature_banks'].keys():
                        if isinstance(output['feature_banks'][k], list):
                            for banks_i in range(len(output['feature_banks'][k])):
                                output['fm'].update({f'F_{k}{banks_i}': output['feature_banks'][k][banks_i].clone().detach()})
                        else:
                            output['fm'].update({f'F_{k}': output['feature_banks'][k].clone().detach()})

                    output['fm'].update({'F_backbone': feature_deform.clone().detach()})
                    output['fm'].update({'F_coarse': feature_coarse.clone().detach()})
        # output['feature_coarse'] = feature_coarse
        output.update({'pixel': [output['pixel']]})
        output['contour_map'] = []
        for gcn_i in range(self.cfg.model.evolve_iters):
            return_vertex_classifier = False
            if self.cfg.model.use_vertex_classifier:
                if gcn_i == self.cfg.model.evolve_iters-1:
                    return_vertex_classifier = True
                if not self.cfg.train.loss_params['vertex_cls']['train_only_final']:
                    return_vertex_classifier = True

            output = self.gcn(output, feature_deform, batch, test_stage=self.test_stage, cfg=self.cfg, return_vertex_classifier=return_vertex_classifier) # contour refine
            if 'py_pred' in output:
                contour_pre = output['py_pred'][-1]
            else:
                contour_pre = output['py'][-1]
            if self.refine_pixel is not None:
                # ✅ .clone()을 사용하여 연산 그래프를 분기합니다.
                # 이렇게 하면 pix_loss의 그래디언트가 py_pred로 안전하게 역전파되면서도,
                # 기존 py_loss의 그래디언트 경로와 충돌하지 않아 RuntimeError를 방지합니다.
                # 이로써 no_grad() 없이도 학습이 가능해져 성능 향상을 기대할 수 있습니다.
                # error 다른 부분으로 해결 - clon() 없애고 돌려서 성능 비교
                refined_pixelmap, contour_map = self.refine_pixel(contour_pre, feature_deform, batch_ind=output['batch_ind'],
                                                                  pre_pixel_map=output['pixel'][-1] if self.cfg.model.ccp_refine_with_pre_pixel else None)
                output['pixel'].append(refined_pixelmap)
                output['contour_map'].append(contour_map)

        # inverse contours for rotation
        if 'test' in batch['meta'] and self.cfg.test.single_rotate_angle is not None:
            for k in ['poly_init', 'poly_coarse']:
                output[k] = inverse_rotate_contours(output[k], angle=self.cfg.test.single_rotate_angle,
                                                         height=self.cfg.data.test_scale[0],
                                                         width=self.cfg.data.test_scale[1])
            for py_i in range(len(output['py'])):
                output['py'][py_i] = inverse_rotate_contours(output['py'][py_i], angle=self.cfg.test.single_rotate_angle, height=self.cfg.data.test_scale[0], width=self.cfg.data.test_scale[1])

        # === Simple flags for 3 stages (init / coarse / final): edit:feat:self-intersection-count:25-08-10 ===
        if (not self.training) and self.cfg.test.track_self_intersection:
            if output['poly_init'].shape[0] > 0:
                sf_init = check_simply_connected(output['poly_init'])
                sf_coarse = check_simply_connected(output['poly_coarse'])
                sf_final = check_simply_connected(output['py'][-1])
                output['simple_flags'] = {'init': sf_init, 'coarse': sf_coarse, 'final': sf_final}

        # post-processing: vertex reduction
        if (not self.training) and self.cfg.model.use_vertex_classifier:
            reduced_contour = self.reduce_vertex(output, min_vertices=self.cfg.test.reduce_min_vertices,
                                                            step=self.cfg.test.reduce_step)
            output['py_reduced'] = reduced_contour
            output['is_simple_reduced'] = check_simply_connected(reduced_contour)

        if (not self.training) and self.cfg.test.check_simple and (len(output['py'][-1]) > 0):
            output['is_simple'] = check_simply_connected(output['py'][-1])

        # Stage 1에서 stage1_train_wh=True일 때만 GT polygon 추가 (학습 시에만)
        if (self.training and 'test' not in batch['meta'] and 
            hasattr(self.cfg.train, 'stage') and self.cfg.train.stage == 1 and
            hasattr(self.cfg.train, 'stage1_train_wh') and self.cfg.train.stage1_train_wh):
            
            from .evolve.utils import collect_training
            ct_01 = output.get('ct_01', batch.get('ct_01', torch.zeros(1, 0, dtype=torch.bool)))
            
            if ct_01.any() and 'img_gt_init_polys' in batch:  # ct_01에 True가 있고 batch에 GT가 있을 때만 처리
                # GT polygons을 batch에서 가져와서 output에 추가 (train_decoder와 gcn 모듈 로직 가져옴)
                output.update({
                    'img_gt_init_polys': collect_training(batch['img_gt_init_polys'], ct_01) * self.cfg.commen.down_ratio,
                    'img_gt_coarse_polys': collect_training(batch['img_gt_coarse_polys'], ct_01) * self.cfg.commen.down_ratio,
                    'img_gt_polys': collect_training(batch['img_gt_polys'], ct_01) * self.cfg.commen.down_ratio
                })

        return output

    def reduce_vertex(self, output, min_vertices=3, step=0.05):
        pred_vertex_xy = output['py'][-1] if 'py' in output else output['py_pred'][-1]
        device = pred_vertex_xy.device

        if 'py_valid_logits' in output:
            py_valid_logits = output['py_valid_logits'][-1] if isinstance(output['py_valid_logits'], list) else output[
                'py_valid_logits']
        else:
            py_valid_logits = None

        if (py_valid_logits is not None) and (py_valid_logits.numel() > 0):
            confidence = F.softmax(py_valid_logits, dim=1)  # (B, C=2, N)

            if self.cfg.test.reduce_apply_adaptive_th:
                valid_scores = confidence[:, 1, ...]  # (B, N)

                # if isinstance(valid_scores, torch.Tensor):
                #     valid_scores = valid_scores.detach().cpu().numpy()
                # if isinstance(pred_vertex_xy, torch.Tensor):
                #     pred_vertex_xy = pred_vertex_xy.detach().cpu().numpy()

                assert pred_vertex_xy.shape[0] == valid_scores.shape[0], "Batch size mismatch"

                reduced_contours = []
                mask_simple = check_simply_connected(pred_vertex_xy)

                for b in range(pred_vertex_xy.shape[0]):
                    contour = pred_vertex_xy[b]  # (V, 2)
                    scores = valid_scores[b]  # (V,)

                    if mask_simple[b]:
                        reduced_contours.append(contour)
                        continue

                    threshold = 0.0
                    reduced = contour

                    while threshold <= 1.:
                        keep_mask = scores >= threshold
                        if keep_mask.sum() < min_vertices:
                            break

                        reduced_candidate = contour[keep_mask]
                        if check_simply_connected(reduced_candidate)[0]:
                            reduced = reduced_candidate
                            break
                        threshold += step

                    reduced_contours.append(reduced)

                return reduced_contours  # list of np.ndarray (V_i, 2)
            else:
                # use fixed threshold
                corner_mask = (confidence[:, 1, ...] >= self.cfg.test.th_score_vertex_cls)  # (B, N) bool
                corner_points = []
                for b in range(pred_vertex_xy.size(0)):
                    mask = corner_mask[b]
                    if mask.sum() >= 3:
                        pts = pred_vertex_xy[b][mask]
                    else:
                        pts = pred_vertex_xy[b]
                    corner_points.append(torch.tensor(pts).to(device=device))

                return corner_points  # list of tensor (V_i, 2)

        else:
            # fallback: just return input
            if isinstance(pred_vertex_xy, torch.Tensor):
                return [v for v in pred_vertex_xy]
            else:
                return list(pred_vertex_xy)

    @torch.no_grad()
    def inference_with_rotation_augmentation(self, x, batch):
        B, C, H, W = x.shape
        angles = [0, 90, 180, 270]
        all_outputs = []

        for angle in angles:
            if angle == 0:
                x_rot = x
            else:
                x_rot = rotate_batch_tensor(x, angle)

            batch_copy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch_copy['meta'] = ['test'] * B  # ensure inference mode
            out = self.forward(x_rot, batch_copy)
            if angle == 0:
                output = out
                output['rotated_py'] = {angle: out['py'] if 'py_reduced' not in out else
                out['py']+out['py_reduced']}
                output['rotated_init'] = {angle: [out['poly_init'], out['poly_coarse']]}
            else:
                output['rotated_py'].update({angle: out['py'] if 'py_reduced' not in out else
                out['py']+out['py_reduced']})
                output['rotated_init'].update({angle: [out['poly_init'], out['poly_coarse']]})

            if 'py_reduced' in out:
                contours = out['py_reduced']
            else:
                contours = out['py'][-1]

            # rotate back
            inv_contours = inverse_rotate_contours(contours, angle, H, W)
            all_outputs.append(inv_contours)

        # now for each sample in batch and each contour, pick the best simple one
        final_contours = []  # length B
        final_rot_id = []

        for b in range(contours.shape[0]):
            best_contour = all_outputs[0][b]  # fallback to 0° if needed
            final_rot_id.append(angles[0])
            for i in range(len(angles)):
                candidate = all_outputs[i][b]
                if check_simply_connected([candidate])[0]:
                    best_contour = candidate
                    final_rot_id[-1] = angles[i]
                    break
            final_contours.append(best_contour)

        output['py_rotate_tta'] = final_contours
        output['id_rotate_tta'] = final_rot_id
        return output

    def net_preprocess(self):
        fix_net_names = []
        if self.cfg.train.fix_dla:
            fix_net_names.append('dla')
        if self.cfg.train.fix_decode:
            fix_net_names.append('train_decoder')
        if self.cfg.train.fix_deform:
            fix_net_names.append('gcn')
        fix_network(self, fix_net_names)

    #====== edit:feat:self-intersection-count:25-08-10 =====
    def _to_poly_list(self, polys):
        if polys is None:
            return []
        if torch.is_tensor(polys):
            # (B,V,2) -> list of (V,2)
            if polys.dim() == 3:
                return [polys[b] for b in range(polys.shape[0])]
            elif polys.dim() == 2:
                return [polys]
            else:
                return []
        if isinstance(polys, (list, tuple)):
            return list(polys)
        t = torch.as_tensor(polys)
        if t.dim() == 3:
            return [t[b] for b in range(t.shape[0])]
        elif t.dim() == 2:
            return [t]
        else:
            return []

    #====== edit:feat:self-intersection-count:25-08-10 =====
    # CCPnet 클래스 내부에 추가
    def _fast_simple_mask_fixedV(self, polys_3d, eps=1e-9, max_bytes_mb=256):
        """
        배치의 모든 폴리곤이 동일한 정점 수 V를 가질 때,
        GPU 벡터화로 '자기교차 없음(simple)' 여부를 빠르게 판정.
        polys_3d: (B, V, 2) float tensor  (cuda 권장)
        반환: (B,) bool tensor  (True=simple, False=자기교차)
        - 인접/연속 간선 교차는 제외
        - 완전 공선(overlap)은 '교차'로 보지 않음(정밀 판정 원하면 fallback에서 처리)
        - 메모리 안전을 위해 B 축을 자동 분할(chunking) 수행
        """
        import math
        assert polys_3d.dim() == 3 and polys_3d.size(-1) == 2, f"Expected (B,V,2), got {tuple(polys_3d.shape)}"

        device = polys_3d.device
        B, V, _ = polys_3d.shape

        # (V,V) 인접 간선 마스크는 B와 무관하므로 한 번만 생성
        idx = torch.arange(V, device=device)
        ii = idx.view(V, 1).expand(V, V)
        jj = idx.view(1, V).expand(V, V)
        adj_2d = (torch.abs(ii - jj) <= 1) | ((ii == 0) & (jj == V - 1)) | ((jj == 0) & (ii == V - 1))
        # adj_2d: (V,V) → chunk에서 (Bc,V,V)로 broadcast

        # 메모리 추정으로 B-chunk 크기 결정
        # 브로드캐스팅 시 (o1,o2,o3,o4): float32 4개 × (Bc*V*V) ≈ 16 bytes/elem
        # inter(bool): (Bc*V*V) ≈ 1 byte/elem  → 대략 17 bytes/elem로 잡고 여유분 반영
        bytes_per_elem = 18.0
        max_bytes = max_bytes_mb * 1024.0 * 1024.0
        denom = max(1.0, bytes_per_elem * V * V)
        Bc = int(max(1, min(B, math.floor(max_bytes / denom))))

        out_simple = torch.ones(B, dtype=torch.bool, device=device)

        def orient(p, q, r):
            # p,q,r: (...,2)
            return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - \
                (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

        for s in range(0, B, Bc):
            e = min(B, s + Bc)
            P = polys_3d[s:e]  # (Bc,V,2)
            Q = torch.roll(P, shifts=-1, dims=1)  # (Bc,V,2)

            a1 = P.unsqueeze(2)  # (Bc,V,1,2)
            a2 = Q.unsqueeze(2)  # (Bc,V,1,2)
            b1 = P.unsqueeze(1)  # (Bc,1,V,2)
            b2 = Q.unsqueeze(1)  # (Bc,1,V,2)

            o1 = orient(a1, a2, b1)
            o2 = orient(a1, a2, b2)
            o3 = orient(b1, b2, a1)
            o4 = orient(b1, b2, a2)

            inter = (o1 * o2 < -eps) & (o3 * o4 < -eps)  # (Bc,V,V)
            inter = inter & (~adj_2d)  # 인접/같은 간선 제외 (broadcast)

            #out_simple[s:e] = ~inter.any(dim=(1, 2))  # (Bc,)
            has_inter = inter.any(dim=2).any(dim=1)  # (Bc,)  ← tuple dim 대신 연속 any
            out_simple[s:e] = ~has_inter

        return out_simple

    # ====== edit:feat:self-intersection-count (3 stages, no topk) ======
    def _stagewise_self_intersection(self, output, return_indices=True):
        use_fast = bool(getattr(self.cfg.test, 'si_fast_gpu', True))
        vtx_stride = int(getattr(self.cfg.test, 'si_vertex_stride', 0))  # 0이면 off

        # 1) 딱 세 단계만 검사: init, coarse, py_last
        stages = []
        if 'poly_init' in output:   stages.append(('init', output['poly_init']))
        if 'poly_coarse' in output: stages.append(('coarse', output['poly_coarse']))
        pys = output.get('py', [])
        if len(pys) > 0:            stages.append(('py_last', pys[-1]))

        # 리스트로 정규화
        def to_list(polys):
            if polys is None: return []
            if torch.is_tensor(polys):
                if polys.dim() == 3: return [polys[b] for b in range(polys.shape[0])]
                if polys.dim() == 2: return [polys]
                return []
            if isinstance(polys, (list, tuple)): return list(polys)
            t = torch.as_tensor(polys)
            return [t[b] for b in range(t.shape[0])] if t.dim() == 3 else ([t] if t.dim() == 2 else [])

        stages = [(n, to_list(t)) for n, t in stages]

        # 배치 크기 B
        B = None
        for _, lst in stages:
            if len(lst) > 0:
                B = len(lst)
                break
        if B is None:
            return {'order': [n for n, _ in stages],
                    'count_new_intersections': {n: 0 for n, _ in stages},
                    'idx_new_intersections': {n: [] for n, _ in stages}}

        # 디바이스
        dev = None
        for _, lst in stages:
            if len(lst) and torch.is_tensor(lst[0]):
                dev = lst[0].device
                break
        if dev is None:
            try:
                dev = next(self.parameters()).device
            except StopIteration:
                dev = torch.device('cpu')

        still_clean = torch.ones(B, dtype=torch.bool, device=dev)
        counts, indices = {}, {}
        first = True

        # (선택) coarse pre-pass 래퍼
        def _coarse_or_fast(polys_3d):
            if vtx_stride and polys_3d.shape[1] >= 3:
                coarse = polys_3d[:, ::vtx_stride, :]
                coarse_simple = self._fast_simple_mask_fixedV(coarse)
                need_refine = ~coarse_simple
                if need_refine.any():
                    refined_simple = self._fast_simple_mask_fixedV(polys_3d[need_refine])
                    coarse_simple[need_refine] = refined_simple
                return coarse_simple
            else:
                return self._fast_simple_mask_fixedV(polys_3d)

        for name, poly_list in stages:
            if len(poly_list) == 0:
                counts[name] = 0
                indices[name] = []
                first = False
                continue

            # 길이 보정
            if len(poly_list) > B:
                poly_list = poly_list[:B]
            elif len(poly_list) < B:
                B = len(poly_list)
                still_clean = still_clean[:B]

            # 동일 V 여부
            sameV = True
            V0 = (poly_list[0].shape[0] if torch.is_tensor(poly_list[0]) else int(np.asarray(poly_list[0]).shape[0]))
            for p in poly_list:
                v = p.shape[0] if torch.is_tensor(p) else int(np.asarray(p).shape[0])
                if v != V0:
                    sameV = False
                    break

            # fast path (고정 V) 또는 fallback
            if use_fast and sameV and (V0 >= 3):
                polys_3d = torch.stack([
                    (p if torch.is_tensor(p) else torch.as_tensor(p)).to(device=dev, dtype=torch.float32)
                    for p in poly_list
                ], dim=0)  # (B,V,2)
                simple_mask = _coarse_or_fast(polys_3d)  # (B,)
            else:
                res = check_simply_connected(poly_list)
                simple_mask = res[0] if isinstance(res, (tuple, list)) else res
                if not torch.is_tensor(simple_mask):
                    simple_mask = torch.as_tensor(simple_mask, dtype=torch.bool)
                else:
                    simple_mask = simple_mask.to(dtype=torch.bool)
                simple_mask = simple_mask.to(still_clean.device)
                if simple_mask.dim() != 1: simple_mask = simple_mask.flatten()
                if simple_mask.numel() > B:
                    simple_mask = simple_mask[:B]
                elif simple_mask.numel() < B:
                    pad = torch.zeros(B - simple_mask.numel(), dtype=torch.bool, device=still_clean.device)
                    simple_mask = torch.cat([simple_mask, pad], dim=0)

            if first:
                counts[name] = 0
                indices[name] = []
                still_clean = still_clean & simple_mask
                first = False
                continue

            new_bad = still_clean & (~simple_mask)
            idx = torch.where(new_bad)[0].tolist()
            counts[name] = len(idx)
            indices[name] = idx if return_indices else []
            still_clean = still_clean & simple_mask

            if not still_clean.any():
                # 남은 스테이지는 0으로 채우고 종료
                for rest_name, _ in stages[stages.index((name, poly_list)) + 1:]:
                    counts[rest_name] = 0
                    indices[rest_name] = []
                break

        return {'order': [n for n, _ in stages],
                'count_new_intersections': counts,
                'idx_new_intersections': indices}


class CCPnetMaskInit(nn.Module):
    """
    변경 핵심:
      1) train_decoder 제거
      2) pixel head 결과를 이진 마스크(0/1)로 만들어 외곽 컨투어를 추출 → poly_init/poly_coarse 생성
      3) 각 컴포넌트(컨투어) 유지 여부는 ct_hm(objectness)로 게이팅
      4) 이후 gcn(EvolutionCCP)·refine_pixel·self-intersection 체크·vertex reduction 등은 기존 로직 유지
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.use_gt_det = cfg.train.use_gt_det     # 유지(사용 안함)
        self.test_stage = cfg.test.test_stage
        self.use_dp = bool(cfg.train.use_dp)
        self.stride = int(cfg.model.init_stride)  # train_decode와 동일 의미
        self.down_sample = int(cfg.commen.down_ratio)  # 최종 이미지 좌표 스케일

        # === backbone & heads (기존과 동일) ===
        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          input_channels=4 if cfg.model.type_add_pixel_mask == 'concat' else 3,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          use_bn_in_head=cfg.model.use_bn_in_head,
                          interpolate_mode='nearest' if cfg.commen.deterministic_mode == "full" else 'bilinear',
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))

        c_in = self.dla.base.channels[self.dla.first_level]
        if self.cfg.model.concat_upper_layer is not None:
            c_in += getattr(self.dla, DICT_dla_module[self.cfg.model.concat_upper_layer.split('_')[0]]).channels[
                int(self.cfg.model.concat_upper_layer.split('_')[-1])]
        # 새로운 concat_multi_layers 채널 수 추가
        if getattr(self.cfg.model, 'concat_multi_layers', None) is not None:
            for layer_name in self.cfg.model.concat_multi_layers:
                _name, _idx = layer_name.split('_')[0], int(layer_name.split('_')[-1])
                c_in += getattr(self.dla, DICT_dla_module[_name]).channels[_idx]
        if self.cfg.model.cat_feature_with_pixelmap and self.cfg.model.cat_include_coarse:
            c_in += cfg.model.heads['pixel']
        if cfg.model.add_grad_feature:
            c_in += 1

        # === train_decoder 삭제됨 ===

        # snake/evolution 모듈(기존 동일)
        self.gcn = EvolutionCCP(
            evolve_iter_num=getattr(cfg.model, 'evolve_iter_num', 1), evolve_stride=cfg.model.evolve_stride,
            ro=cfg.commen.down_ratio, with_img_idx=cfg.model.with_img_idx, refine_kernel_size=cfg.model.refine_kernel_size,
            in_featrue_dim=c_in, use_vertex_classifier=cfg.model.use_vertex_classifier,
            channel_pixel=1 if cfg.model.ccp_deform_pixel_norm=='argmax' else cfg.model.heads['pixel'],
            c_out_proj=cfg.model.ccp_dim_out_proj, with_proj=cfg.model.ccp_with_proj, num_vertex=cfg.model.points_per_poly,
            pixel_norm_type=cfg.model.ccp_deform_pixel_norm, vtx_cls_common_prediction_type=cfg.model.vtx_cls_common_prediction_type,
            vtx_cls_kernel_size=cfg.model.vtx_cls_kernel_size, cfg=cfg,
            gcn_weight_sharing=getattr(cfg.model, 'gcn_weight_sharing', True),
            use_3x3_feature=getattr(cfg.model, 'use_3x3_feature', False),
            feature_3x3_mode=getattr(cfg.model, 'feature_3x3_mode', 'flatten')
        )

        # 픽셀 리파인(기존 동일)
        if cfg.model.use_refine_pixel:
            refine_dim_in = c_in
            dim_pixel = 1 if cfg.model.ccp_refine_pixel_input_norm == 'argmax' else cfg.model.heads['pixel']
            refine_dim_in += (1+dim_pixel if cfg.model.ccp_refine_with_pre_pixel else 1)
            self.refine_pixel = RefinePixel(
                dim_out=cfg.model.heads['pixel'], down_ratio=cfg.commen.down_ratio,
                num_layers=cfg.model.refine_pixel_param.get('num_layers', 2),
                dim_list=cfg.model.refine_pixel_param.get('dim_list', [256]),
                kernel_list=cfg.model.refine_pixel_param.get('kernel_list', [3,1]),
                dim_in=refine_dim_in, input_norm_type=cfg.model.ccp_refine_pixel_input_norm,
                reduce_memory=cfg.model.contour_map_down_sample, refine_as_residual=cfg.model.ccp_refine_pixel_as_residual,
                convert_map_down_ratio=cfg.model.contour_map_down_ratio,
                module_structure=cfg.model.refine_pixel_param.get('module_structure', None),
                contour_map_shape=[cfg.data.input_h//cfg.commen.down_ratio, cfg.data.input_w//cfg.commen.down_ratio],
                deterministic=cfg.commen.deterministic_mode
            )
        else:
            self.refine_pixel = None

        self.net_preprocess()

    def _make_small_circle(self, cy, cx, r, Nv, H, W):
        """중심(cy,cx) 기준 반지름 r 픽셀짜리 원형 다각형(Nv개) 생성 (이미지 경계 클립)"""
        ang = np.linspace(0, 2 * np.pi, Nv, endpoint=False, dtype=np.float32)
        xs = np.clip(cx + r * np.cos(ang), 0, W - 1)
        ys = np.clip(cy + r * np.sin(ang), 0, H - 1)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def _iou_with_poly(self, H, W, poly_a_xy, poly_b_xy):
        """두 다각형(poly_a, poly_b)의 픽셀 IoU 계산 (둘 다 (N,2) float, 좌표계: (x,y))."""
        import numpy as np, cv2
        if poly_a_xy is None or poly_b_xy is None: return 0.0
        ma = np.zeros((H, W), dtype=np.uint8)
        mb = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(ma, [poly_a_xy.astype(np.int32)], 1)
        cv2.fillPoly(mb, [poly_b_xy.astype(np.int32)], 1)
        inter = (ma & mb).sum()
        union = (ma | mb).sum()
        return float(inter) / float(union + 1e-6)
    
    def _validate_coords_in_bounds(self, coords, H, W, task_name=''):
        """
        좌표가 경계 내에 있는지 체크 (ccp_maskinit task만)
        수정하지 않고 경고만 출력
        Returns: 유효한 좌표 마스크
        """
        if 'ccp_maskinit' not in self.cfg.commen.task:
            # 다른 task는 기존 방식 유지
            return torch.ones_like(coords[..., 0], dtype=torch.bool)
        
        out_of_bounds_x = (coords[..., 0] < 0) | (coords[..., 0] >= W)
        out_of_bounds_y = (coords[..., 1] < 0) | (coords[..., 1] >= H)
        out_of_bounds = out_of_bounds_x | out_of_bounds_y
        
        if out_of_bounds.any():
            num_out = out_of_bounds.sum().item()
            total = out_of_bounds.numel()
            print(f"[WARNING][ccp_maskinit] {task_name}: {num_out}/{total} points out of bounds (H={H}, W={W})")
            
            # 디버깅용 상세 정보
            if num_out < 10:  # 소수만 있을 때 상세 출력
                x_min = coords[..., 0].min().item()
                x_max = coords[..., 0].max().item()
                y_min = coords[..., 1].min().item()
                y_max = coords[..., 1].max().item()
                print(f"  X range: [{x_min:.1f}, {x_max:.1f}], Y range: [{y_min:.1f}, {y_max:.1f}]")
        
        return ~out_of_bounds

    def _align_polygon_vertices_tensor(self, pred_poly, gt_poly):
        """
        ccp_maskinit 전용: GT와 Prediction polygon의 vertex index 정렬 (Tensor 버전)
        Gradient 유지를 위해 torch tensor로 처리
        
        Args:
            pred_poly: (N, 2) prediction polygon tensor
            gt_poly: (N, 2) ground truth polygon tensor
        Returns:
            aligned_pred_poly: (N, 2) 정렬된 prediction polygon tensor
        """
        if 'ccp_maskinit' not in self.cfg.commen.task:
            return pred_poly
        
        import torch
        
        # detach()로 인덱스 계산만 수행 (gradient 영향 없음)
        pred_detached = pred_poly.detach()
        gt_detached = gt_poly.detach()
        
        # GT vertex 0과 가장 가까운 prediction vertex 찾기
        gt_v0 = gt_detached[0:1]  # (1, 2)
        distances = torch.norm(pred_detached - gt_v0, dim=1)  # (N,)
        best_start_idx = torch.argmin(distances).item()
        
        # Forward 방향으로만 정렬 (GT가 일관된 방향으로 정렬되어 있음)
        indices = torch.arange(len(pred_poly), device=pred_poly.device)
        rolled_indices = torch.roll(indices, -best_start_idx)
        aligned_pred = pred_poly[rolled_indices]  # gradient 유지
        
        return aligned_pred
    
    def _match_gt_prediction_pairs(self, pred_polys, gt_polys, batch_ind=None, max_distance=2.5):
        """
        GT와 Prediction polygon들을 center distance 기반으로 매칭
        Args:
            pred_polys: (N, V, 2) prediction polygons  
            gt_polys: (M, V, 2) ground truth polygons
            batch_ind: (N,) 각 prediction의 batch index (선택적)
            max_distance: 최대 허용 center distance (feature map 좌표계, image 좌표계에서는 ~10픽셀)
        Returns:
            matched_pred_polys: (K, V, 2) 매칭된 prediction polygons
            matched_gt_polys: (K, V, 2) 매칭된 ground truth polygons
            matched_batch_ind: (K,) 매칭된 batch indices (batch_ind가 주어진 경우)
            matched_gt_indices: (K,) 원본 GT 인덱스 (keypoints_mask 재배열용)
        """
        if 'ccp_maskinit' not in self.cfg.commen.task:
            if batch_ind is not None:
                # 매칭 없이 원본 그대로 반환, 인덱스도 순서대로
                gt_indices = torch.arange(len(gt_polys), device=gt_polys.device)
                return pred_polys, gt_polys, batch_ind, gt_indices
            return pred_polys, gt_polys
            
        import torch
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        
        # GT는 이미지별로 순서대로 배치되어 있음 (collect_training 결과)
        # Prediction도 이미지별로 순서대로 배치되어 있어야 함
        # 하지만 실제로는 서로 다른 순서일 수 있으므로 이미지별 매칭 필요
        
        if batch_ind is None:
            # Fallback to global matching when batch_ind is not available
            # Fallback: global 매칭 (기존 방식)
            pred_centers = pred_polys.mean(dim=1)  # (N, 2)
            gt_centers = gt_polys.mean(dim=1)      # (M, 2)
            
            distances = torch.cdist(pred_centers, gt_centers)  # (N, M)
            distances_np = distances.detach().cpu().numpy()
            
            pred_indices, gt_indices = linear_sum_assignment(distances_np)
            
            valid_matches = []
            for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                if distances_np[pred_idx, gt_idx] <= max_distance:
                    valid_matches.append((pred_idx, gt_idx))
        else:
            # 이미지별 매칭 수행
            batch_ind_np = batch_ind.detach().cpu().numpy()
            unique_imgs = np.unique(batch_ind_np)
            
            all_matched_pred_indices = []
            all_matched_gt_indices = []
            
            # 각 이미지별로 매칭 수행
            for img_idx in unique_imgs:
                # 해당 이미지의 prediction 인덱스들
                pred_mask = (batch_ind_np == img_idx)
                pred_img_indices = np.where(pred_mask)[0]
                
                if len(pred_img_indices) == 0:
                    continue
                
                # GT는 collect_training으로 이미지별 순서대로 배치됨
                # Prediction과 동일한 이미지에 속하는 GT 인덱스 찾기
                # 간단한 방법: GT도 이미지별로 순서대로 배치되어 있다고 가정하고
                # 현재까지 처리된 prediction 수만큼 GT 인덱스를 할당
                num_pred_in_img = len(pred_img_indices)
                gt_start_idx = sum(len(np.where(batch_ind_np == prev_img)[0]) for prev_img in unique_imgs if prev_img < img_idx)
                gt_img_indices = list(range(gt_start_idx, min(gt_start_idx + num_pred_in_img, len(gt_polys))))
                
                if len(gt_img_indices) == 0:
                    continue
                
                # 해당 이미지 내에서 center distance 기반 매칭
                pred_img_polys = pred_polys[pred_img_indices]
                gt_img_polys = gt_polys[gt_img_indices]
                
                pred_centers = pred_img_polys.mean(dim=1)  # (num_pred, 2)
                gt_centers = gt_img_polys.mean(dim=1)      # (num_gt, 2)
                
                distances = torch.cdist(pred_centers, gt_centers)  # (num_pred, num_gt)
                distances_np = distances.detach().cpu().numpy()
                
                # Hungarian 매칭
                img_pred_indices, img_gt_indices = linear_sum_assignment(distances_np)
                
                # 거리 제한 확인 후 유효한 매칭만 추가
                for p_idx, g_idx in zip(img_pred_indices, img_gt_indices):
                    if distances_np[p_idx, g_idx] <= max_distance:
                        all_matched_pred_indices.append(pred_img_indices[p_idx])
                        all_matched_gt_indices.append(gt_img_indices[g_idx])
            
            valid_matches = list(zip(all_matched_pred_indices, all_matched_gt_indices))
        
        if len(valid_matches) == 0:
            # 매칭 실패 시 빈 텐서 반환
            empty_shape = (0, pred_polys.shape[1], pred_polys.shape[2])
            empty_pred = torch.empty(empty_shape, device=pred_polys.device)
            empty_gt = torch.empty(empty_shape, device=gt_polys.device)
            empty_gt_indices = torch.empty((0,), dtype=torch.long, device=gt_polys.device)
            if batch_ind is not None:
                empty_batch_ind = torch.empty((0,), dtype=batch_ind.dtype, device=batch_ind.device)
                return empty_pred, empty_gt, empty_batch_ind, empty_gt_indices
            return empty_pred, empty_gt, empty_gt_indices
        
        # 매칭된 polygon들 추출
        matched_pred_indices = [match[0] for match in valid_matches]
        matched_gt_indices = [match[1] for match in valid_matches]
        
        matched_pred_polys = pred_polys[matched_pred_indices]
        matched_gt_polys = gt_polys[matched_gt_indices]
        
        # GT 인덱스를 텐서로 변환
        matched_gt_indices_tensor = torch.tensor(matched_gt_indices, device=gt_polys.device, dtype=torch.long)
        
        # Matching statistics can be logged here if needed
        
        if batch_ind is not None:
            matched_batch_ind = batch_ind[matched_pred_indices]
            return matched_pred_polys, matched_gt_polys, matched_batch_ind, matched_gt_indices_tensor
        
        return matched_pred_polys, matched_gt_polys, matched_gt_indices_tensor
    
    def _unified_coord_transform(self, coords, from_space='feature', to_space='image'):
        """
        학습/테스트 모두에서 사용하는 통합 좌표 변환 (ccp_maskinit task만)
        Args:
            coords: 좌표 텐서
            from_space: 'feature' (104x104) or 'image' (416x416)
            to_space: 'feature' or 'image'
        """
        if 'ccp_maskinit' not in self.cfg.commen.task:
            # 다른 task는 기존 방식 유지
            if from_space == 'feature' and to_space == 'image':
                return coords * self.cfg.commen.down_ratio
            elif from_space == 'image' and to_space == 'feature':
                return coords / self.cfg.commen.down_ratio
            else:
                return coords
        
        # ccp_maskinit: 일관된 좌표 변환
        if from_space == to_space:
            return coords
        
        down_ratio = self.cfg.commen.down_ratio
        
        if from_space == 'feature' and to_space == 'image':
            # 피처맵 -> 이미지 좌표
            result = coords * down_ratio
            # 변환 후 validate (clamp 하지 않음)
            if coords.numel() > 0:
                H, W = self.cfg.data.input_h, self.cfg.data.input_w
                self._validate_coords_in_bounds(result, H, W, 
                                               task_name=f'Transform {from_space}->{to_space}')
            return result
        elif from_space == 'image' and to_space == 'feature':
            # 이미지 -> 피처맵 좌표
            result = coords / down_ratio
            # 변환 후 validate (clamp 하지 않음)
            if coords.numel() > 0:
                H = self.cfg.data.input_h // down_ratio
                W = self.cfg.data.input_w // down_ratio
                self._validate_coords_in_bounds(result, H, W,
                                               task_name=f'Transform {from_space}->{to_space}')
            return result
        else:
            raise ValueError(f"[ccp_maskinit] Invalid space conversion: {from_space} -> {to_space}")
    
    def _pick_best_contour_for_gt(self, contours, gt_poly_px, H, W, iou_th=0.1):
        """GT 폴리곤과 IoU 최대인 컨투어 선택(임계 미만이면 포함/최단거리로 대체)."""
        if not contours: return None
        # 1) IoU 최대
        best_iou, best = -1.0, None
        for c in contours:
            if c is None or len(c) < 3: continue
            iou = self._iou_with_poly(H, W, gt_poly_px, c)
            if iou > best_iou:
                best_iou, best = iou, c
        if best_iou >= iou_th:
            return best

        # 2) 포함 여부(inside) 우선
        cx, cy = gt_poly_px[:, 0].mean(), gt_poly_px[:, 1].mean()
        best2, best_d = None, 1e9
        for c in contours:
            cc = c.reshape(-1, 1, 2).astype(np.float32)
            d = cv2.pointPolygonTest(cc, (float(cx), float(cy)), True)
            if d >= 0:  # inside
                return c
            ad = abs(d)
            if ad < best_d:
                best_d, best2 = ad, c
        return best2

    def _train_inits_from_gt(self, batch, output, cnn_feature, get_feature=False):
        """
        TRAIN:
          - GT 폴리곤(gt_polys): (B, max_len, Nv_gt, 2)
          - pixel head → 이진 마스크 → 이미지별 외곽 컨투어
          - 각 GT와 가장 잘 맞는 컨투어 선택(IoU>th; 없으면 inside/최단거리 기반) → Nv_out로 리샘플
          - 반환: poly_init_flat (N, Nv_out, 2), batch_ind (N,)
            * N = 배치 내 유효 GT 총 개수 (ct_01 == True 개수)
        """
        device = cnn_feature.device

        # --- 0) GT 폴리곤 가져오기: (B, max_len, Nv_gt, 2)
        gt_key = 'img_gt_polys'
        gt_polys = batch[gt_key].to(device=device, dtype=torch.float32)  # (B, max_len, Nv_gt, 2)
        B, max_len, Nv_gt, _ = gt_polys.shape

        # 모델 설정과 GT 정점 수가 다르면, init은 GT Nv에 맞추는 기존 정책 유지
        Nv_out = int(self.cfg.commen.points_per_poly)
        if Nv_out != Nv_gt:
            Nv_out = Nv_gt

        # --- 1) pixel map & mask/contours (이미지별)
        pixel_map = output['pixel']  # (B, C, Hp, Wp)  (train 경로에선 tensor)
        Hp, Wp = pixel_map.shape[-2:]
        # print(f"[DEBUG] pixel_map.shape[-2:] : {pixel_map.shape[-2:]}")
        
        # ✅ mask 생성 (gradient path는 pixel loss를 통해 유지됨)
        mask01 = self._pixel_to_binary_mask(pixel_map)  # (B,1,Hp,Wp)  {0,1}
        
        # contour 추출을 위한 numpy 변환 (이 부분은 불가피하게 gradient 단절)
        # 하지만 pixel loss가 별도로 계산되므로 pixel head 학습은 여전히 가능
        with torch.no_grad():
            mask_np = (mask01.detach().cpu().numpy() * 255).astype(np.uint8)
            contours_all = [self._find_outer_contours(mask_np[b, 0]) for b in range(B)]  # list per image

        # --- 2) 유효 GT 마스크 (B, max_len)
        if 'ct_01' in batch:
            valid = batch['ct_01'].to(device).bool()  # (B, max_len)
        else:
            # fallback: 좌표 합으로 대충 유효 판단 (전부 0이면 무효)
            valid = (gt_polys.abs().sum(dim=(-2, -1)) > 0)

        # --- 3) GT는 이미 pixel head 좌표계 (104x104)
        gt_px = gt_polys.clone()
        
        # ccp_maskinit: clamp 대신 validate 사용
        if 'ccp_maskinit' in self.cfg.commen.task:
            valid_gt_coords = self._validate_coords_in_bounds(gt_px, Hp, Wp, task_name='GT coords')
            # 경계 밖 좌표가 있으면 경고만 출력하고 계속 진행
            # 실제 매칭 시 문제가 되면 자연스럽게 걸러짐
        else:
            # 다른 task는 기존 clamp 유지
            gt_px[..., 0].clamp_(min=0, max=Wp - 1)
            gt_px[..., 1].clamp_(min=0, max=Hp - 1)

        # --- 4) 매칭/리샘플 (매칭된 GT들만 추적)
        # ✅ IoU 임계값을 더 낮춰서 빠른 매칭 허용
        iou_th = float(getattr(self.cfg.train, 'init_match_iou_th', 0.01))  # 0.05 → 0.01
        r_fallback = int(getattr(self.cfg.train, 'init_circle_radius', 3))

        init_list = []
        batch_inds = []
        matched_gt_indices = []  # ✅ 성공적으로 매칭된 (b, k) 인덱스 추적
        
        for b in range(B):
            if not valid[b].any():
                continue
            
            # 유효한 슬롯 인덱스(k들)
            ks = torch.nonzero(valid[b], as_tuple=False).squeeze(1).tolist()
            batch_matched = 0
            used_contours = set()  # ✅ 이미 사용된 컨투어 인덱스 추적
            
            # ✅ GT-Contour 매칭 점수 계산하여 최적 매칭 찾기
            gt_contour_pairs = []
            for k in ks:
                # ✅ ccp_maskinit: gradient 보존을 위해 tensor 형태로 유지
                if 'ccp_maskinit' in self.cfg.commen.task:
                    gt_bk_tensor = gt_px[b, k]  # tensor 형태로 유지
                    gt_bk = gt_bk_tensor.detach().cpu().numpy()  # 매칭 계산용으로만 변환
                else:
                    gt_bk = gt_px[b, k].detach().cpu().numpy()
                
                best_contour_idx = None
                best_score = float('inf')  # distance는 작을수록 좋음
                
                # ccp_maskinit task의 경우 center 거리 기반 매칭
                if 'ccp_maskinit' in self.cfg.commen.task:
                    # GT contour center 계산
                    gt_center = np.mean(gt_bk, axis=0)  # (2,) [x, y]
                    
                    # 최대 거리 임계값 가져오기 (ccp_maskinit 전용)
                    max_distance = getattr(self.cfg.train, 'max_center_distance_maskinit', 50.0)
                    
                    for c_idx, contour in enumerate(contours_all[b]):
                        if c_idx in used_contours:  # 이미 사용된 컨투어는 스킵
                            continue
                        if len(contour) < 3:
                            continue
                        
                        # Predicted contour center 계산
                        pred_center = np.mean(contour, axis=0)  # (2,) [x, y]
                        
                        # Euclidean distance
                        distance = np.linalg.norm(gt_center - pred_center)
                        
                        # 최대 거리 임계값 체크 (ccp_maskinit에만 적용)
                        if distance < max_distance and distance < best_score:
                            best_score = distance
                            best_contour_idx = c_idx
                
                else:
                    # 다른 task들은 기존 IoU 방식 유지
                    for c_idx, contour in enumerate(contours_all[b]):
                        if c_idx in used_contours:  # 이미 사용된 컨투어는 스킵
                            continue
                        if len(contour) < 3:
                            continue
                            
                        iou = self._iou_with_poly(Hp, Wp, gt_bk, contour)
                        if iou > best_score and iou >= iou_th:
                            best_score = iou
                            best_contour_idx = c_idx
                
                if best_contour_idx is not None:
                    gt_contour_pairs.append((k, best_contour_idx, best_score))
            
            # ✅ ccp_maskinit는 거리 오름차순, 다른 task는 IoU 내림차순으로 정렬
            if 'ccp_maskinit' in self.cfg.commen.task:
                gt_contour_pairs.sort(key=lambda x: x[2])  # distance 오름차순 (가까운 것부터)
            else:
                gt_contour_pairs.sort(key=lambda x: x[2], reverse=True)  # IoU 내림차순
            
            # ✅ 1:1 매칭 수행
            for k, c_idx, score in gt_contour_pairs:
                if c_idx not in used_contours:
                    contour = contours_all[b][c_idx]
                    poly_xy = self._resample_closed_poly(contour.astype(np.float32), Nv_out)
                    init_list.append(torch.from_numpy(poly_xy).to(device=device, dtype=torch.float32))
                    batch_inds.append(b)
                    matched_gt_indices.append((b, k))
                    used_contours.add(c_idx)
                    batch_matched += 1
            
            # ✅ 매칭에 실패한 GT는 학습에서 제외 (fallback 제거)

        # --- 5) (N,Nv,2) pixel grid → 이미지 좌표로 복원
        if len(init_list) > 0:
            # ✅ [수정] down_ratio(s) 곱셈을 제거합니다.
            # 이 함수는 이제 일관되게 피처맵 좌표계(0~104)의 컨투어를 반환하며,
            # 이미지 좌표계로의 변환은 forward() 메서드에서 한 번만 수행됩니다.
            poly_init_flat = torch.stack(init_list, dim=0)
            batch_ind = torch.tensor(batch_inds, device=device, dtype=torch.long)  # (N,)
        else:
            poly_init_flat = torch.empty(0, Nv_out, 2, device=device, dtype=torch.float32)
            batch_ind = torch.empty(0, dtype=torch.long, device=device)

        # --- 6) 매칭된 GT들만 반영한 ct_01 관련 텐서 생성
        # ✅ 매칭 성공한 GT들만 유효하다고 마킹
        ct_01_vec = torch.zeros_like(valid, dtype=torch.bool, device=device)  # (B, max_len)
        for b, k in matched_gt_indices:
            ct_01_vec[b, k] = True

        # ct_img_idx: (B, K) long, 유효한 위치에 b 인덱스, 나머지는 -1
        ct_img_idx = torch.full((B, max_len), -1, dtype=torch.long, device=device)
        for b in range(B):
            if ct_01_vec[b].any():
                ct_img_idx[b, ct_01_vec[b]] = b

        # ✅ 매칭 통계 출력
        total_gt = valid.sum().item()
        matched_gt = len(matched_gt_indices)

        return poly_init_flat, batch_ind, ct_01_vec, ct_img_idx

    # ------------------------- 새 경로: pixel→mask→contour -------------------------
    @staticmethod
    def _sigmoid_mask(pixel_logits, th=0.5):
        """ pixel head 채널 수가 1인 이진 로지트 → 시그모이드 후 threshold """
        prob = torch.sigmoid(pixel_logits)  # (B,1,H,W)
        return (prob >= th).to(pixel_logits.dtype)

    @staticmethod
    def _softmax_fg(pixel_logits):
        """ 다중 채널인 경우: argmax가 0(배경) 외 클래스면 1, 아니면 0 """
        # 가정: ch=0이 배경. 다르면 cfg로 조정 필요
        prob = F.softmax(pixel_logits, dim=1)        # (B,C,H,W)
        pred = torch.argmax(prob, dim=1, keepdim=True)  # (B,1,H,W)
        return (pred != 0).to(pixel_logits.dtype)

    def _pixel_to_binary_mask(self, pixel_map):
        """
        pixel head 출력을 이진 마스크로 변환.
        - 채널=1: sigmoid≥cfg.test.pixel_th
        - 채널>1: argmax!=0 foreground
        """
        th = float(getattr(self.cfg.test, 'pixel_th', 0.5))
        if pixel_map.shape[1] == 1:
            return self._sigmoid_mask(pixel_map, th)
        else:
            return self._softmax_fg(pixel_map)

    @staticmethod
    def _resample_closed_poly(poly_xy, M):
        """
        poly_xy: (N,2) ndarray(float32), 닫힌 외곽선(시작/끝 중복 허용/불허 모두 OK)
        반환: (M,2) 등간격 호선 길이 재표본
        """
        if poly_xy.shape[0] < 3:
            # 최소 삼각형으로 확장(안전가드)
            return np.pad(poly_xy, ((0, max(0, 3 - poly_xy.shape[0])), (0,0)), mode='edge')[:M]

        # 닫힘 보장
        is_closed = np.allclose(poly_xy[0], poly_xy[-1])
        pts = poly_xy if is_closed else np.vstack([poly_xy, poly_xy[0:1]])

        seg = np.sqrt(((pts[1:] - pts[:-1])**2).sum(1))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        L = s[-1] if s[-1] > 0 else 1.0
        s_new = np.linspace(0, L, num=M+1)[:-1]  # 마지막=첫점 중복 제거
        x = np.interp(s_new, s, pts[:,0])
        y = np.interp(s_new, s, pts[:,1])
        return np.stack([x, y], axis=1).astype(np.float32)

    def _find_outer_contours(self, mask_01):
        """
        mask_01: (H,W) uint8
        반환: list of ndarray (Ni, 2)  좌표계: (x,y) = (col,row)
        """
        cnts, _ = cv2.findContours(mask_01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in cnts:
            if c.shape[0] < 3:  # too small
                continue
            # c: (n,1,2) in (x,y)
            polys.append(c[:,0,:].astype(np.float32))
        return polys

    def _filter_mask_by_cc(self, mask_01, min_area=200, topk=None):
        # mask_01: (H,W) uint8 in {0,1}
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_01, connectivity=8)
        # stats: [label, x, y, w, h, area]
        keep = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if topk is not None and len(keep) > topk:
            # 큰 면적 TopK만
            keep = sorted(keep, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:topk]
        filt = np.zeros_like(mask_01, dtype=np.uint8)
        for i in keep:
            filt[labels == i] = 1
        return filt

    def _gate_by_ct_objectness(self, polys, ct_hm_up, min_score, min_area=20, max_keep=None):
        """
        polys: list of (Ni,2) ndarray in image scale
        ct_hm_up: (H,W) torch float [0,1]  (upsampled)
        각 폴리곤의 bounding box/centroid에서의 ct score로 게이팅(간단/빠름)
        """
        H, W = ct_hm_up.shape
        keep = []
        for p in polys:
            # 면적/둘레 작은 노이즈 제거
            area = float(cv2.contourArea(p.reshape(-1,1,2)))
            if area < float(min_area):
                continue
            cx = np.clip(int(np.round(p[:,0].mean())), 0, W-1)
            cy = np.clip(int(np.round(p[:,1].mean())), 0, H-1)
            score = float(ct_hm_up[cy, cx].item())
            if score >= min_score:
                keep.append(p)
        # 간단히 면적 큰 순으로 top-k
        if max_keep is not None and len(keep) > max_keep:
            keep = sorted(keep, key=lambda q: -cv2.contourArea(q.reshape(-1,1,2)))[:max_keep]
        return keep

    # === 1) 게이팅: 폴리곤 내부 max(ct) 기반으로 선별 & 정렬(top-K) ===
    def _gate_by_ct_objectness_maxins(self, polys, ct_hm_up, min_score, topk=None, min_area=20):
        """
        polys: list of (Ni,2) float32 ndarray, image-scale (x,y)
        ct_hm_up: (H,W) torch float [0..1]  (upsampled ct heatmap)
        반환: kept_polys(list), kept_scores(list)
        - 각 폴리곤 내부 픽셀의 max(ct) >= min_score 인 것만 유지
        - max(ct) 내림차순 정렬 후 top-K 선택 (K 부족하면 있는 만큼)
        """
        H, W = ct_hm_up.shape
        kept, scores = [], []

        # 로컬 numpy 뷰 (mask 만들 때만 사용)
        ct_np = ct_hm_up.detach().cpu().numpy()

        for p in polys:
            # 너무 작은 건 제거
            area = float(cv2.contourArea(p.reshape(-1, 1, 2)))
            if area < float(min_area):
                continue

            # bbox 클리핑
            pi = np.round(p).astype(np.int32)
            x0 = max(int(np.floor(pi[:, 0].min())), 0)
            y0 = max(int(np.floor(pi[:, 1].min())), 0)
            x1 = min(int(np.ceil(pi[:, 0].max())), W - 1)
            y1 = min(int(np.ceil(pi[:, 1].max())), H - 1)
            if x1 <= x0 or y1 <= y0:
                continue

            # bbox-sized mask
            mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
            poly_roi = (pi - np.array([x0, y0], dtype=np.int32)).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [poly_roi], 1)

            roi = ct_np[y0:y1 + 1, x0:x1 + 1]
            if mask.any():
                maxv = float(roi[mask == 1].max())
            else:
                maxv = 0.0

            if maxv >= float(min_score):
                kept.append(p)
                scores.append(maxv)

        if len(kept) == 0:
            return [], []

        # max score 내림차순 정렬
        order = np.argsort(-np.asarray(scores))
        if topk is not None:
            order = order[:min(int(topk), len(order))]
        kept_sorted = [kept[i] for i in order]
        scores_sorted = [scores[i] for i in order]
        # print(f"[DEBUG] len(kept_sorted): {len(kept_sorted)}, scores_sorted: {len(scores_sorted)}")
        return kept_sorted, scores_sorted

    def _masks_to_init_polys(
            self, pixel_logits, ct_hm, Nv, min_ct_score, min_area, topk
    ):
        """
        반환:
          polys_t: (Ntot, Nv, 2) float32
          inds_t:  (Ntot,) long   # 각 폴리곤이 어떤 이미지에서 왔는지
          ct_01_vec: (B, K) bool
          ct_img_idx: (B, K) long
          ct_peaks_yx: (B, K, 2) long  # (y,x), 채워지지 않은 곳은 -1
          ct_01_map: (B,1,H,W) same dtype as pixel head (옵션)
        """
        B, _, H, W = pixel_logits.shape
        # print(f"[DEBUG/pixel_logits] B,H,W : {B},{H},{W}")
        device = pixel_logits.device
        K = int(topk) if topk is not None else 0  # K=topk, 없으면 아래서 max로 정함

        # 1) pixel → binary mask
        mask01 = self._pixel_to_binary_mask(pixel_logits)  # (B,1,H,W)
        mask_np = (mask01.detach().cpu().numpy() * 255).astype(np.uint8)

        # NEW: 선필터 (작은 blob 제거, 과도 검출 억제)
        min_area = int(min_area)  # 기존 인자 재사용
        pre_topk = int(topk) if topk is not None else None  # 너무 크면 여기서도 살짝 제한
        for b in range(mask_np.shape[0]):
            mask_np[b, 0] = self._filter_mask_by_cc(mask_np[b, 0], min_area=min_area,
                                                    topk=pre_topk * 3 if pre_topk else None)

        # 2) ct upsample to (H,W)
        ct_prob = torch.sigmoid(ct_hm)
        if ct_prob.shape[-2:] != (H, W):
            ct_prob_up = F.interpolate(ct_prob, size=(H, W), mode="bilinear", align_corners=False)
        else:
            ct_prob_up = ct_prob
        ct_prob_up = ct_prob_up[:, 0]  # (B,H,W)
        # print(f"ct_prob_up : {ct_prob_up.shape}")

        # ✅ DEBUG: pixel-ct 상관관계 시각화 및 통계 분석 (주석 처리)
        # import matplotlib.pyplot as plt
        # import os
        # vis_dir = f"{self.cfg.commen.result_dir}/debug_vis"
        # os.makedirs(vis_dir, exist_ok=True)
        
        # 상관관계 통계 수집
        # correlations = []

        all_polys = []
        all_inds = []
        per_img_counts = []
        per_img_peaks = []  # list of list[(y,x)]
        for b in range(B):
            # 외곽 컨투어
            polys = self._find_outer_contours(mask_np[b, 0])  # list of (Ni,2) float32
            # print(f"mask_np : {mask_np.shape}")

            # ✅ DEBUG: 시각화 및 통계 분석 (주석 처리)
            # if b == 0 and len(polys) > 0:
            #     # 1) 시각화
            #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            #     
            #     # Pixel mask
            #     mask_vis = mask_np[b, 0]
            #     axes[0].imshow(mask_vis, cmap='gray')
            #     axes[0].set_title('Pixel Mask')
            #     
            #     # CT heatmap
            #     ct_vis = ct_prob_up[b].detach().cpu().numpy()
            #     im = axes[1].imshow(ct_vis, cmap='hot', vmin=0, vmax=1)
            #     axes[1].set_title(f'CT Heatmap (max: {ct_vis.max():.3f})')
            #     plt.colorbar(im, ax=axes[1])
            #     
            #     # Overlay + detected contours
            #     axes[2].imshow(mask_vis, cmap='gray', alpha=0.7)
            #     axes[2].imshow(ct_vis, cmap='hot', alpha=0.5, vmin=0, vmax=1)
            #     
            #     # 검출된 컨투어와 점수 표시
            #     for i, p in enumerate(polys):
            #         # 컨투어 그리기
            #         axes[2].plot(p[:, 0], p[:, 1], 'lime', linewidth=2)
            #         
            #         # 중심점에서의 ct 점수 계산
            #         cx = int(np.round(p[:, 0].mean()))
            #         cy = int(np.round(p[:, 1].mean()))
            #         if 0 <= cx < W and 0 <= cy < H:
            #             score = ct_vis[cy, cx]
            #             axes[2].plot(cx, cy, 'red', marker='o', markersize=8)
            #             axes[2].text(cx, cy-10, f'{score:.3f}', color='white', fontsize=8)
            #     
            #     axes[2].set_title('Overlay + Contours + CT Scores')
            #     
            #     plt.tight_layout()
            #     plt.savefig(f"{vis_dir}/mask_ct_correlation_batch{b}.png", dpi=150)
            #     plt.close()

            # # 2) 통계 분석
            # mask_binary = (mask_np[b, 0] > 0).astype(np.float32)
            # ct_values = ct_prob_up[b].detach().cpu().numpy()
            # 
            # if mask_binary.sum() > 0:
            #     avg_ct_in_mask = ct_values[mask_binary > 0].mean()
            #     avg_ct_outside_mask = ct_values[mask_binary == 0].mean()
            #     
            #     correlations.append({
            #         'avg_ct_in_mask': avg_ct_in_mask,
            #         'avg_ct_outside_mask': avg_ct_outside_mask,
            #         'ratio': avg_ct_in_mask / (avg_ct_outside_mask + 1e-6)
            #     })

            # max-inside(ct) 점수로 게이팅 + 내림차순 정렬 + top-K
            polys_kept, scores = self._gate_by_ct_objectness_maxins(
                polys, ct_prob_up[b], min_score=min_ct_score,
                topk=topk, min_area=min_area
            )
            
            # ✅ ct_score 필터링 전 원본 contour 저장 (옵션)
            if getattr(self.cfg.test, 'save_pixel_initial_contours', False):
                # 원본 contour들을 별도로 저장 (리샘플링 후)
                polys_unfiltered = []
                for p in polys:  # 필터링 전 모든 contour
                    if len(p) >= 3:  # 최소 3점 이상
                        pr = self._resample_closed_poly(p, Nv)  # (Nv,2)
                        pr[:, 0] = np.clip(pr[:, 0], 0, W-1)
                        pr[:, 1] = np.clip(pr[:, 1], 0, H-1)
                        polys_unfiltered.append(pr)
                
                # 배치별로 저장할 키 생성
                if b == 0:  # 첫 번째 배치에서 키 초기화
                    if not hasattr(self, '_pixel_initial_contours_batch'):
                        self._pixel_initial_contours_batch = {}
                
                self._pixel_initial_contours_batch[b] = polys_unfiltered


            # ✅ DEBUG: 필터링 전후 비교 (주석 처리)
            # print(f"[DEBUG] Batch {b}: Contours before={len(polys)}, after={len(polys_kept)}")
            # if len(polys) > 0:
            #     all_scores = []
            #     for p in polys:
            #         cx = int(np.round(p[:, 0].mean()))
            #         cy = int(np.round(p[:, 1].mean()))
            #         if 0 <= cx < W and 0 <= cy < H:
            #             score = ct_prob_up[b][cy, cx].item()
            #             all_scores.append(score)
            #     
            #     if all_scores:
            #         print(f"  Score range: {min(all_scores):.4f} - {max(all_scores):.4f}")
            #         print(f"  Scores < {min_ct_score}: {sum(1 for s in all_scores if s < min_ct_score)}/{len(all_scores)}")
            #         print(f"  min_ct_score threshold: {min_ct_score}")

            # 등간격 리샘플 & 피크 위치(y,x) 저장
            peaks_b = []
            if len(polys_kept) > 0:
                ct_np = ct_prob_up[b].detach().cpu().numpy()
                for p in polys_kept:
                    # 리샘플
                    pr = self._resample_closed_poly(p, Nv)  # (Nv,2)
                    # ✅ 리샘플 후 좌표 범위 제한
                    pr[:, 0] = np.clip(pr[:, 0], 0, W-1)
                    pr[:, 1] = np.clip(pr[:, 1], 0, H-1)
                    # print(f"pr : {pr.max()}")
                    all_polys.append(pr)
                    all_inds.append(b)

                    # 피크 좌표 계산(폴리곤 내부 max-ct)
                    pi = np.round(p).astype(np.int32)
                    x0 = max(int(np.floor(pi[:, 0].min())), 0)
                    y0 = max(int(np.floor(pi[:, 1].min())), 0)
                    x1 = min(int(np.ceil(pi[:, 0].max())), W - 1)
                    y1 = min(int(np.ceil(pi[:, 1].max())), H - 1)
                    if x1 > x0 and y1 > y0:
                        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
                        poly_roi = (pi - np.array([x0, y0], dtype=np.int32)).reshape(-1, 1, 2)
                        cv2.fillPoly(mask, [poly_roi], 1)
                        roi = ct_np[y0:y1 + 1, x0:x1 + 1]
                        if mask.any():
                            yy, xx = np.where(mask == 1)
                            flat = np.argmax(roi[yy, xx])
                            cy, cx = int(y0 + yy[flat]), int(x0 + xx[flat])
                            peaks_b.append((cy, cx))
                            continue
                    peaks_b.append((-1, -1))  # fallback
            per_img_counts.append(len(peaks_b))
            per_img_peaks.append(peaks_b)

        # 3) 직사각 텐서로 패킹(= dataloader 방식)
        if K == 0:
            K = max(per_img_counts) if len(per_img_counts) > 0 else 0

        ct_01_vec = torch.zeros((B, K), dtype=torch.bool, device=device)
        ct_img_idx = torch.zeros((B, K), dtype=torch.long, device=device)
        ct_peaks_yx = torch.full((B, K, 2), -1, dtype=torch.long, device=device)  # 변수명을 yx로 변경하여 명확하게

        for b in range(B):
            m = min(per_img_counts[b], K)
            if m > 0:
                ct_01_vec[b, :m] = True
                ct_img_idx[b, :m] = b
                if per_img_peaks[b]:
                    yx = np.array(per_img_peaks[b][:m], dtype=np.int64)
                    ct_peaks_yx[b, :m, 0] = int(0) + torch.as_tensor(yx[:, 0], device=device)
                    ct_peaks_yx[b, :m, 1] = int(0) + torch.as_tensor(yx[:, 1], device=device)

        # 4) 스패셜 맵(옵션): top-K 피크 지점(혹은 작은 원)만 1
        make_map = bool(getattr(self.cfg.test, "export_ct01_map", False))
        if make_map and K > 0:
            ct01_map = torch.zeros((B, 1, H, W), dtype=pixel_logits.dtype, device=device)
            rad = int(getattr(self.cfg.test, "ct_peak_radius", 0))
            for b in range(B):
                m = ct_01_vec[b].sum().item()
                for j in range(m):
                    y, x = int(ct_peaks_yx[b, j, 0].item()), int(ct_peaks_yx[b, j, 1].item())
                    if y < 0 or x < 0:
                        continue
                    if rad <= 0:
                        ct01_map[b, 0, y, x] = 1
                    else:
                        # 간단한 디스크 찍기(정밀/속도 트레이드오프)
                        yy_min = max(0, y - rad)
                        yy_max = min(H, y + rad + 1)
                        xx_min = max(0, x - rad)
                        xx_max = min(W, x + rad + 1)
                        ct01_map[b, 0, yy_min:yy_max, xx_min:xx_max] = 1
        else:
            ct01_map = None

        # ✅ DEBUG: 최종 상관관계 통계 출력 (주석 처리)
        # if correlations:
        #     print(f"[DEBUG] Pixel-CT correlation stats:")
        #     print(f"  Avg CT in mask: {np.mean([c['avg_ct_in_mask'] for c in correlations]):.4f}")
        #     print(f"  Avg CT outside: {np.mean([c['avg_ct_outside_mask'] for c in correlations]):.4f}")
        #     print(f"  Ratio: {np.mean([c['ratio'] for c in correlations]):.4f}")
        #     print(f"  Total batches analyzed: {len(correlations)}")

        # 5) 최종 텐서 변환
        if len(all_polys) == 0:
            polys_t = torch.empty(0, Nv, 2, device=device, dtype=torch.float32)
            inds_t = torch.empty(0, device=device, dtype=torch.long)
        else:
            polys_t = torch.from_numpy(np.stack(all_polys, axis=0)).to(device=device, dtype=torch.float32)
            inds_t = torch.as_tensor(all_inds, device=device, dtype=torch.long)
            
            # ✅ ccp_maskinit에서 테스트/validation 시에도 좌표 범위 제한 - 접힘 현상 방지
            if polys_t.numel() > 0:
                polys_t[..., 0] = torch.clamp(polys_t[..., 0], min=0, max=W-1)
                polys_t[..., 1] = torch.clamp(polys_t[..., 1], min=0, max=H-1)

        B, H, W = ct_prob_up.shape
        # print(f"[DEBUG/in mask_to_init] B,H,W : {B}x{H}x{W}")
        device = ct_prob_up.device
        K = ct_01_vec.shape[1] if ct_01_vec.numel() > 0 else 0

        if K > 0:
            yx = ct_peaks_yx.clone()  # (B,K,2), long - 이제 변수명이 정확함: [y, x] 순서
            valid = (yx[..., 0] >= 0) & (yx[..., 1] >= 0)
            # 선형 인덱스: y * W + x
            lin_idx = (yx[..., 0] * W + yx[..., 1])  # (B,K)
            # Set invalid indices to 0 to avoid negative indices in gather
            lin_idx[~valid] = 0
            ct_flat = ct_prob_up.reshape(B, -1)

            ct_scores_vec = torch.zeros((B, K), dtype=ct_prob_up.dtype, device=device)
            # gather로 (B,K) 위치 점수 뽑기
            gathered = torch.gather(ct_flat, 1, lin_idx.to(torch.long))
            ct_scores_vec[valid] = gathered[valid]

            # (N,)으로 납작하게 — 폴리곤 텐서와 같은 순서
            ct_scores_n = ct_scores_vec[ct_01_vec].unsqueeze(1)  # (N,1)
        else:
            ct_scores_vec = None
            ct_scores_n = torch.empty(0, 1, dtype=ct_prob_up.dtype, device=device)

        return polys_t, inds_t, ct_01_vec, ct_img_idx, ct_peaks_yx, ct01_map, ct_scores_n, ct_scores_vec

    # -----------------------------------------------------------------------------

    def forward(self, x, batch=None):
        if self.cfg.model.type_add_pixel_mask == 'concat':
            input = torch.cat([x, batch['pixel_gt'].to(x.device).unsqueeze(1)], dim=1)
        else:
            input = x
        if 'test' in batch['meta'] and self.cfg.test.single_rotate_angle is not None:
            input = torch.rot90(input, k=self.cfg.test.single_rotate_angle//90, dims=(2, 3))

        # backbone + heads
        output, cnn_feature, feature_banks = self.dla(input, save_banks=self.cfg.test.get_featuremap)

        # feature_deform 구성(기존 동일)
        if self.cfg.model.cat_feature_with_pixelmap:
            pixel_map = F.softmax(output['pixel'], dim=1) if self.cfg.model.cat_feature_normalized else output['pixel']
            if pixel_map.shape[2:] == cnn_feature.shape[2:]:
                add_feature = pixel_map
            else:
                add_feature = F.interpolate(pixel_map, size=cnn_feature.shape[2:], mode='nearest')
            feature_deform = torch.cat((cnn_feature, add_feature.detach() if self.cfg.model.cut_grad_add_feature else add_feature), 1)
        else:
            feature_deform = cnn_feature

        # feature_deform_coarse = feature_deform if self.cfg.model.cat_include_coarse else cnn_feature  # (호환 유지)

        if self.training:
            # ===== TRAIN: GT 매칭 기반 init (기존 train_decode 로직 호환) =====
            poly_init, batch_ind, ct_01, ct_img_idx = \
                self._train_inits_from_gt(batch, output, feature_deform, get_feature=False)

            # evolve가 compact 텐서를 직접 쓰도록 키 제공 (길이 불일치 방지)
            # print(f"[DEBUG/Training] poly_init : {poly_init.shape}")
            
            # ✅ ccp_maskinit에서 학습 시 좌표 범위 체크
            if poly_init.numel() > 0:
                pixel_H, pixel_W = output['pixel'].shape[-2:]
                
                if 'ccp_maskinit' in self.cfg.commen.task:
                    # ccp_maskinit: clamp 대신 validate만 수행
                    self._validate_coords_in_bounds(poly_init, pixel_H, pixel_W, 
                                                   task_name='Training poly_init')
                else:
                    # 다른 task는 기존 clamp 유지
                    poly_init[..., 0] = torch.clamp(poly_init[..., 0], min=0, max=pixel_W-1)
                    poly_init[..., 1] = torch.clamp(poly_init[..., 1], min=0, max=pixel_H-1)
                
                # ✅ device 일치 확인 및 수정
                target_device = output['pixel'].device
                if poly_init.device != target_device:
                    poly_init = poly_init.to(target_device)
                if batch_ind.device != target_device:
                    batch_ind = batch_ind.to(target_device)
            
            # 좌표 변환은 아래에서 task별로 처리됨
            
            # batch_ind는 매칭 과정에서 업데이트될 수 있음
            if 'batch_ind' not in output:
                output['batch_ind'] = batch_ind  # (N,)
            output['ct_01'] = ct_01
            
            # GT 좌표도 통합 변환 사용
            if 'ccp_maskinit' in self.cfg.commen.task:
                gt_coarse = collect_training(batch['img_gt_coarse_polys'], ct_01)
                gt_init = collect_training(batch['img_gt_init_polys'], ct_01)
                gt_final = collect_training(batch['img_gt_polys'], ct_01)
                
                # ✅ ccp_maskinit에서 GT-Prediction 매칭 후 vertex alignment 적용
                if poly_init.numel() > 0 and gt_final.numel() > 0:
                    # GT-Prediction center-based 매칭 수행 (batch_ind도 함께)
                    matched_poly_init, matched_gt_final, matched_batch_ind, matched_gt_indices = self._match_gt_prediction_pairs(
                        poly_init, gt_final, batch_ind)
                    
                    if matched_poly_init.shape[0] > 0:
                        # 매칭된 쌍들에 대해 vertex alignment 수행
                        aligned_poly_init = matched_poly_init.clone()
                        for i in range(matched_poly_init.shape[0]):
                            aligned_poly_init[i] = self._align_polygon_vertices_tensor(
                                matched_poly_init[i], matched_gt_final[i]
                            )
                        
                        # 결과 저장
                        output['poly_init'] = self._unified_coord_transform(aligned_poly_init, 'feature', 'image')
                        output['py_pred'] = [self._unified_coord_transform(aligned_poly_init.clone(), 'feature', 'image')]
                        
                        # GT도 매칭된 순서로 저장
                        output['img_gt_polys'] = self._unified_coord_transform(matched_gt_final, 'feature', 'image')
                        
                        # batch_ind도 매칭된 순서로 업데이트
                        output['batch_ind'] = matched_batch_ind
                        
                        # ✅ keypoints_mask도 매칭된 순서로 재배열
                        if 'keypoints_mask' in batch and ct_01 is not None:
                            original_keypoints_mask = collect_training(batch['keypoints_mask'], ct_01)
                            output['matched_keypoints_mask'] = original_keypoints_mask[matched_gt_indices]
                    else:
                        # 매칭 실패 시 기존 방식 사용
                        output['poly_init'] = self._unified_coord_transform(poly_init, 'feature', 'image')
                        output['py_pred'] = [self._unified_coord_transform(poly_init.clone(), 'feature', 'image')]
                        output['img_gt_polys'] = self._unified_coord_transform(gt_final, 'feature', 'image')
                else:
                    # 빈 데이터인 경우 기존 방식 사용
                    output['poly_init'] = self._unified_coord_transform(poly_init, 'feature', 'image')
                    output['py_pred'] = [self._unified_coord_transform(poly_init.clone(), 'feature', 'image')]
                
                # GT 데이터 설정 (매칭으로 이미 설정된 경우 덮어쓰지 않음)
                if 'img_gt_polys' not in output:
                    output['img_gt_polys'] = self._unified_coord_transform(gt_final, 'feature', 'image')
                output.update({
                    'img_gt_coarse_polys': self._unified_coord_transform(gt_coarse, 'feature', 'image'),
                    'img_gt_init_polys': self._unified_coord_transform(gt_init, 'feature', 'image')
                })
            else:
                # 다른 task는 기존 방식 유지
                output['poly_init'] = poly_init * self.cfg.commen.down_ratio
                output['py_pred'] = [poly_init.clone() * self.cfg.commen.down_ratio]
                output.update({
                    'img_gt_coarse_polys': collect_training(batch['img_gt_coarse_polys'], ct_01) * self.cfg.commen.down_ratio,
                    'img_gt_init_polys': collect_training(batch['img_gt_init_polys'], ct_01) * self.cfg.commen.down_ratio,
                    'img_gt_polys': collect_training(batch['img_gt_polys'], ct_01) * self.cfg.commen.down_ratio
                })
            # print(f"[DEBUG] in network : img_gt_polys : {output['img_gt_polys'].max()}")
        else:
            with torch.no_grad():
                # === 바뀐 부분: pixel→mask→contour 로 init/coarse 생성 ===
                Nv = int(self.cfg.commen.points_per_poly)
                min_ct_score = float(self.cfg.test.ct_score)
                min_area = float(getattr(self.cfg.test, 'min_component_area', 20))
                topk = int(getattr(self.cfg.test, 'topK', 100))

                # 여기서 사용하는 pixel map은 gcn/refine_pixel 경로와 동일 소스
                pixel_logits = output['pixel']  # (B,C,H,W)
                ct_hm = output['ct_hm']         # (B,1,hc,wc)

                poly_init, batch_ind, ct_01_vec, ct_img_idx, ct_peaks_yx, ct01_map, ct_scores_n, ct_scores_vec = self._masks_to_init_polys(
                    pixel_logits, ct_hm, Nv, min_ct_score=min_ct_score, min_area=min_area, topk=topk
                )
                # print(f"[DEBUG] len(poly_init): {len(poly_init)}, batch_ind: {batch_ind.shape}")
                # coarse는 우선 init과 동일(원하면 여기서 smoothing/approxPolyDP 적용 가능)
                poly_coarse = poly_init.clone()

                # gcn이 요구하는 키 구성
                # ccp_maskinit: 테스트에서도 통합 좌표 변환 사용
                if 'ccp_maskinit' in self.cfg.commen.task:
                    output['poly_init'] = self._unified_coord_transform(poly_init, 'feature', 'image')
                    output['py'] = [self._unified_coord_transform(poly_coarse, 'feature', 'image')]
                else:
                    # 다른 task는 기존 방식 유지
                    output['poly_init'] = poly_init * self.cfg.commen.down_ratio
                    output['py'] = [poly_coarse * self.cfg.commen.down_ratio]
                output['batch_ind'] = batch_ind
                # print(f"[DEBUG/Test] poly_init,min{poly_init.min()},max{poly_init.max()}")

                output['ct_01_vec'] = ct_01_vec  # (B, K) bool
                output['ct_img_idx'] = ct_img_idx  # (B, K) long
                output['ct_peaks_yx'] = ct_peaks_yx  # (B, K, 2) long

                # ✅ pixel map에서 추출한 ct_score 필터링 전 initial contour 저장
                if getattr(self.cfg.test, 'save_pixel_initial_contours', False) and hasattr(self, '_pixel_initial_contours_batch'):
                    output['pixel_initial_contours'] = self._pixel_initial_contours_batch
                    # 메모리 정리
                    delattr(self, '_pixel_initial_contours_batch')

                if ct01_map is not None:
                    output['ct_01'] = ct01_map
                else:
                    output['ct_01'] = ct_01_vec
                # print(f"poly_init : {poly_init.shape}, ct_01: map{ct01_map.shape}, vec{ct_01_vec.shape} / batch_ind : {batch_ind.shape}")

                device = poly_init.device
                N = poly_init.shape[0]

                if N == 0:
                    # 빈 배치 처리: task별로 최소 열수만 맞춰줌
                    output['detection'] = torch.empty(0, 2, dtype=torch.float32, device=device)  # [score, label]
                else:
                    sc = ct_scores_n.to(torch.float32)  # (N,1)

                    # ⚠️ 라벨은 "단일 인덱스"로 (one-hot 금지)
                    # 여기선 전부 foreground=1로 둠. 멀티클래스면 argmax 로직으로 교체.
                    label = torch.zeros(N, 1, dtype=torch.float32, device=device)

                    # === task별 detection 포맷 ===
                    detection = torch.cat([sc, label], dim=1)  # (N,2)

                    # 출력 키
                    output['detection'] = detection

        # 픽셀 스텝 리스트화(기존과 동일)
        output.update({'pixel': [output['pixel']]})
        output['contour_map'] = []

        # === snake/evolution + optional refine_pixel (기존 동일) ===
        for gcn_i in range(self.cfg.model.evolve_iters):
            return_vertex_classifier = False
            if self.cfg.model.use_vertex_classifier:
                if gcn_i == self.cfg.model.evolve_iters-1:
                    return_vertex_classifier = True
                if not self.cfg.train.loss_params['vertex_cls']['train_only_final']:
                    return_vertex_classifier = True

            output = self.gcn(output, feature_deform, batch, test_stage=self.test_stage,
                              cfg=self.cfg, return_vertex_classifier=return_vertex_classifier)

            contour_pre = output['py_pred'][-1] if 'py_pred' in output else output['py'][-1]
            # if contour_pre.numel() > 0:
                # print(f"[DEBUG] contour_pre : min{contour_pre.min()}, max{contour_pre.max()}")

            if self.refine_pixel is not None:
                refined_pixelmap, contour_map = self.refine_pixel(
                    contour_pre, feature_deform, batch_ind=output['batch_ind'],
                    pre_pixel_map=output['pixel'][-1] if self.cfg.model.ccp_refine_with_pre_pixel else None
                )
                output['pixel'].append(refined_pixelmap)
                output['contour_map'].append(contour_map)

        # inverse contours for rotation (기존 유지)
        if 'test' in batch['meta'] and self.cfg.test.single_rotate_angle is not None:
            for k in ['poly_init']:
                output[k] = inverse_rotate_contours(output[k], angle=self.cfg.test.single_rotate_angle,
                                                    height=self.cfg.data.test_scale[0], width=self.cfg.data.test_scale[1])
            for py_i in range(len(output['py'])):
                output['py'][py_i] = inverse_rotate_contours(
                    output['py'][py_i], angle=self.cfg.test.single_rotate_angle,
                    height=self.cfg.data.test_scale[0], width=self.cfg.data.test_scale[1]
                )

        # 간단 연결성 플래그(기존 유지)
        if (not self.training) and self.cfg.test.track_self_intersection:
            if output['poly_init'].shape[0] > 0:
                sf_init = check_simply_connected(output['poly_init'])
                # sf_coarse = check_simply_connected(output['poly_coarse'])
                sf_final = check_simply_connected(output['py'][-1])
                output['simple_flags'] = {'init': sf_init, 'final': sf_final}

        # vertex reduction(기존 유지)
        if (not self.training) and self.cfg.model.use_vertex_classifier:
            reduced_contour = self.reduce_vertex(output, min_vertices=self.cfg.test.reduce_min_vertices,
                                                 step=self.cfg.test.reduce_step)
            output['py_reduced'] = reduced_contour
            output['is_simple_reduced'] = check_simply_connected(reduced_contour)

        if (not self.training) and self.cfg.test.check_simple and (len(output['py'][-1]) > 0):
            output['is_simple'] = check_simply_connected(output['py'][-1])

        # feature bank 제공(기존 유지)
        if (not self.training) and self.cfg.test.get_featuremap:
            output['feature_banks'] = feature_banks
            output['cnn_feature'] = feature_deform
            output['fm'] = {}
            for k in feature_banks.keys():
                if isinstance(feature_banks[k], list):
                    for banks_i in range(len(feature_banks[k])):
                        output['fm'].update({f'F_{k}{banks_i}': feature_banks[k][banks_i].clone().detach()})
                else:
                    output['fm'].update({f'F_{k}': feature_banks[k].clone().detach()})
            output['fm'].update({'F_backbone': feature_deform.clone().detach()})
            # coarse 특징을 별도로 쓰지 않아도 일단 동일하게 채워둠(호환)
            output['fm'].update({'F_coarse': feature_deform.clone().detach()})

        return output

    # ===== 기존 부가 기능(수정 없이 이식) =================================================
    def reduce_vertex(self, output, min_vertices=3, step=0.05):
        # (사용자 제공 코드 그대로)
        pred_vertex_xy = output['py'][-1] if 'py' in output else output['py_pred'][-1]
        device = pred_vertex_xy.device

        if 'py_valid_logits' in output:
            py_valid_logits = output['py_valid_logits'][-1] if isinstance(output['py_valid_logits'], list) else output['py_valid_logits']
        else:
            py_valid_logits = None

        if (py_valid_logits is not None) and (py_valid_logits.numel() > 0):
            confidence = F.softmax(py_valid_logits, dim=1)  # (B,2,N)

            if self.cfg.test.reduce_apply_adaptive_th:
                valid_scores = confidence[:, 1, ...]  # (B,N)
                assert pred_vertex_xy.shape[0] == valid_scores.shape[0], "Batch size mismatch"

                reduced_contours = []
                mask_simple = check_simply_connected(pred_vertex_xy)

                for b in range(pred_vertex_xy.shape[0]):
                    contour = pred_vertex_xy[b]
                    scores = valid_scores[b]
                    if mask_simple[b]:
                        reduced_contours.append(contour)
                        continue
                    threshold = 0.0
                    reduced = contour
                    while threshold <= 1.:
                        keep_mask = scores >= threshold
                        if keep_mask.sum() < min_vertices:
                            break
                        reduced_candidate = contour[keep_mask]
                        if check_simply_connected(reduced_candidate)[0]:
                            reduced = reduced_candidate
                            break
                        threshold += step
                    reduced_contours.append(reduced)
                return reduced_contours
            else:
                corner_mask = (confidence[:, 1, ...] >= self.cfg.test.th_score_vertex_cls)
                corner_points = []
                for b in range(pred_vertex_xy.size(0)):
                    mask = corner_mask[b]
                    pts = pred_vertex_xy[b][mask] if mask.sum() >= 3 else pred_vertex_xy[b]
                    corner_points.append(torch.tensor(pts).to(device=device))
                return corner_points
        else:
            if isinstance(pred_vertex_xy, torch.Tensor):
                return [v for v in pred_vertex_xy]
            else:
                return list(pred_vertex_xy)

    @torch.no_grad()
    def inference_with_rotation_augmentation(self, x, batch):
        # (사용자 제공 코드 그대로)
        B, C, H, W = x.shape
        angles = [0, 90, 180, 270]
        all_outputs = []
        for angle in angles:
            if angle == 0:
                x_rot = x
            else:
                x_rot = rotate_batch_tensor(x, angle)
            batch_copy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch_copy['meta'] = ['test'] * B
            out = self.forward(x_rot, batch_copy)
            if angle == 0:
                output = out
                output['rotated_py'] = {angle: out['py'] if 'py_reduced' not in out else out['py']+out['py_reduced']}
                output['rotated_init'] = {angle: [out['poly_init']]}
            else:
                output['rotated_py'].update({angle: out['py'] if 'py_reduced' not in out else out['py']+out['py_reduced']})
                output['rotated_init'].update({angle: [out['poly_init']]})
            contours = out['py_reduced'] if 'py_reduced' in out else out['py'][-1]
            inv_contours = inverse_rotate_contours(contours, angle, H, W)
            all_outputs.append(inv_contours)

        final_contours = []
        final_rot_id = []
        for b in range(contours.shape[0]):
            best_contour = all_outputs[0][b]
            final_rot_id.append(angles[0])
            for i in range(len(angles)):
                candidate = all_outputs[i][b]
                if check_simply_connected([candidate])[0]:
                    best_contour = candidate
                    final_rot_id[-1] = angles[i]
                    break
            final_contours.append(best_contour)

        output['py_rotate_tta'] = final_contours
        output['id_rotate_tta'] = final_rot_id
        return output

    def net_preprocess(self):
        fix_net_names = []
        if self.cfg.train.fix_dla:
            fix_net_names.append('dla')
        # train_decoder는 없음
        if self.cfg.train.fix_deform:
            fix_net_names.append('gcn')
        fix_network(self, fix_net_names)


class CCPnetPyramid(CCPnet):
    """
    Pyramid deform schedule:
      - low-res feature (1/(down_ratio*2)): init+coarse @ 16 verts
      - densify +2/edge -> 48 verts, deform x2 @ mid-res
      - densify +1/edge -> 96 verts, deform x1 @ mid-res
    Pixel map: extracted at mid-res (same as cnn_feature resolution).
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.low_down_ratio = 2
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          input_channels=4 if cfg.model.type_add_pixel_mask == 'concat' else 3,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          use_bn_in_head=cfg.model.use_bn_in_head,
                          interpolate_mode='nearest' if cfg.commen.deterministic_mode == "full" else 'bilinear',
                          head_stage_map={'pixel':'mid', 'ct_hm':'low', 'wh':'low'},
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))
        # --- low-res 16-vertex용 decoder (동일) ---
        self.train_decoder_low = Decode(
            num_point=cfg.commen.init_points_per_poly,
            init_stride=cfg.model.init_stride,
            coarse_stride=cfg.model.coarse_stride,
            down_sample=cfg.commen.down_ratio * self.low_down_ratio,
            min_ct_score=cfg.test.ct_score,
            c_in=self._infer_c_in_for_lowdecoder(),
            num_point_each_step=cfg.commen.points_per_poly_steps,
            with_img_idx=cfg.model.with_img_idx,
            refine_kernel_size=cfg.model.refine_kernel_size,
            use_dp=cfg.train.use_dp,
            use_trans_feature=cfg.model.refine_use_trans_feature,
            cat_include_coarse=cfg.model.cat_include_coarse
        )

        # --- 단계별 횟수: config에서 읽기 ---
        # 옵션 A) dict: cfg.model.deform_iters_per_stage = {"stage1": 2, "stage2": 1}
        # 옵션 B) list: cfg.model.deform_iters_per_stage = [2, 1]  # [for 48, for 96]
        di = getattr(cfg.model, "deform_iters_per_stage", {"stage1": 2, "stage2": 1})
        if isinstance(di, dict):
            iters_1 = int(di.get("stage1", 2))
            iters_2 = int(di.get("stage2", 1))
        else:
            iters_1 = int(di[0]) if len(di) > 0 else 2
            iters_2 = int(di[1]) if len(di) > 1 else 1

        common_kwargs = dict(
            evolve_iter_num=getattr(cfg.model, 'evolve_iter_num', 1),
            evolve_stride=cfg.model.evolve_stride,
            ro=cfg.commen.down_ratio,
            with_img_idx=cfg.model.with_img_idx,
            refine_kernel_size=cfg.model.refine_kernel_size,
            in_featrue_dim=self._infer_c_in_for_lowdecoder(),
            use_vertex_classifier=cfg.model.use_vertex_classifier,
            channel_pixel=1 if cfg.model.ccp_deform_pixel_norm == 'argmax' else cfg.model.heads['pixel'],
            c_out_proj=cfg.model.ccp_dim_out_proj,
            with_proj=cfg.model.ccp_with_proj,
            num_vertex=cfg.model.points_per_poly,
            pixel_norm_type=cfg.model.ccp_deform_pixel_norm,
            vtx_cls_common_prediction_type=cfg.model.vtx_cls_common_prediction_type,
            vtx_cls_kernel_size=cfg.model.vtx_cls_kernel_size,
            cfg=cfg,  # ✅ cfg 추가
            gcn_weight_sharing=getattr(cfg.model, 'gcn_weight_sharing', True)
        )

        # --- 단계별 snake 풀 (weight sharing 없음) ---
        self.gcn_stage1 = nn.ModuleList([EvolutionCCP(**common_kwargs) for _ in range(max(iters_1, 0))])
        self.gcn_stage2 = nn.ModuleList([EvolutionCCP(**common_kwargs) for _ in range(max(iters_2, 0))])

        # freeze 처리
        self._net_preprocess_no_share()

    def _net_preprocess_no_share(self):
        # 기존 fix_network와 병행: ModuleList까지 확실히 얼리기
        fix_net_names = []
        if self.cfg.train.fix_dla:
            fix_net_names.append('dla')
        if self.cfg.train.fix_decode:
            fix_net_names += ['train_decoder', 'train_decoder_low']
        fix_network(self, fix_net_names)

        if self.cfg.train.fix_deform:
            for m in [self.gcn_stage1, self.gcn_stage2]:
                for sub in m.modules():
                    for p in sub.parameters():
                        p.requires_grad = False

    # 기존 __init__에서의 c_in 계산을 재사용하기 위한 헬퍼
    def _infer_c_in_for_lowdecoder(self):
        c_in = self.dla.base.channels[self.dla.first_level]
        if self.cfg.model.concat_upper_layer is not None:
            c_in += getattr(self.dla, DICT_dla_module[self.cfg.model.concat_upper_layer.split('_')[0]]).channels[
                int(self.cfg.model.concat_upper_layer.split('_')[-1])]
        if self.cfg.model.cat_feature_with_pixelmap and self.cfg.model.cat_include_coarse:
            c_in += self.cfg.model.heads['pixel']
        if self.cfg.model.add_grad_feature:
            c_in += 1
        return c_in

    @staticmethod
    def _densify_uniform(contours: torch.Tensor, k_insert_per_edge: int) -> torch.Tensor:
        """
        contours: (B, V, 2)  — polygon is closed (last edge V-1 -> 0)
        k_insert_per_edge: e.g., 2 -> insert 2 points at 1/3, 2/3 along each edge
        returns: (B, V*(k+1), 2)
        """
        assert contours.dim() == 3 and contours.size(-1) == 2
        B, V, _ = contours.shape
        nxt = torch.roll(contours, shifts=-1, dims=1)  # (B,V,2) next vertex for each v_i
        seg = nxt - contours  # (B,V,2)

        # steps includes 0.0 to keep original vertex, but excludes 1.0 (handled by next v start)
        steps = [0.0] + [ (j/(k_insert_per_edge+1)) for j in range(1, k_insert_per_edge+1) ]  # length k+1
        pts = []
        for s in steps:
            pts.append(contours + s * seg)  # (B,V,2)
        out = torch.stack(pts, dim=2)  # (B,V,k+1,2)
        out = out.reshape(B, V*(k_insert_per_edge+1), 2)  # interleave per-edge
        return out

    def forward(self, x, batch=None):
        # ===== 1) 입력/백본 =====
        if self.cfg.model.type_add_pixel_mask == 'concat':
            inp = torch.cat([x, batch['pixel_gt'].to(x.device).unsqueeze(1)], dim=1)
        else:
            inp = x

        # 단일 회전 테스트 옵션 유지
        if 'test' in batch['meta'] and self.cfg.test.single_rotate_angle is not None:
            inp = torch.rot90(inp, k=self.cfg.test.single_rotate_angle//90, dims=(2, 3))

        # get feature map
        output, cnn_feature_mid, cnn_feature_low, feature_banks = self.dla(inp, save_banks=self.cfg.test.get_featuremap, return_stage_backbones=True)

        # ===== 2) mid-res feature (원래 cnn_feature) + pixel 융합 =====
        if self.cfg.model.cat_feature_with_pixelmap:
            pixel_map = F.softmax(output['pixel'], dim=1) if self.cfg.model.cat_feature_normalized else output['pixel']
            if pixel_map.shape[2:] == cnn_feature_mid.shape[2:]:
                add_feature = pixel_map
            else:
                add_feature = F.interpolate(pixel_map, size=cnn_feature_mid.shape[2:], mode='nearest')
            feature_mid = torch.cat((cnn_feature_mid, add_feature.detach() if self.cfg.model.cut_grad_add_feature else add_feature), 1)
        else:
            feature_mid = cnn_feature_mid

        # ===== 4) init/coarse on low-res with 16 verts =====
        if 'test' not in batch['meta']:
            feature_coarse, output = self.train_decoder_low(batch, cnn_feature_low, output, is_training=True)
        else:
            with torch.no_grad():
                ignore = (self.test_stage == 'init')
                feature_coarse, output = self.train_decoder_low(
                    batch, cnn_feature_low, output, is_training=False,
                    ignore_gloabal_deform=ignore, get_feature=self.cfg.test.get_featuremap
                )
                if (not self.training) and self.cfg.test.get_featuremap:
                    output['feature_banks'] = feature_banks
                    output['cnn_feature'] = cnn_feature_mid
                    output['fm'] = {}
                    for k in output['feature_banks'].keys():
                        if isinstance(output['feature_banks'][k], list):
                            for i in range(len(output['feature_banks'][k])):
                                output['fm'].update({f'F_{k}{i}': output['feature_banks'][k][i].clone().detach()})
                        else:
                            output['fm'].update({f'F_{k}': output['feature_banks'][k].clone().detach()})
                    output['fm'].update({'F_backbone': feature_mid.clone().detach()})
                    output['fm'].update({'F_coarse': feature_coarse.clone().detach()})

        # 출력 통일
        output.update({'pixel': [output['pixel']]})
        output['contour_map'] = []

        # ==== 현재 contour(16) 가져오기 ====
        base_contour = output['poly_coarse']  # (B,16,2) 가정

        # ===== 5) 16 -> 48 : +2/edge 균일 삽입, mid-res deform x2 =====
        # 16 -> 48 (+2/edge) 후, stage48의 ModuleList 길이만큼 반복
        num_p_insert_1 = 2
        py_48 = self._densify_uniform(base_contour, k_insert_per_edge=num_p_insert_1)  # (B,48,2)
        if self.training:
            output.setdefault('py_pred', []).append(py_48)
            output.setdefault('py_keys', []).append('up_1')
            init = prepare_training(output, batch, ro=self.cfg.commen.down_ratio,
                                    num_points=self.cfg.commen.init_points_per_poly * num_p_insert_1, set_py=False)
            output['batch_ind'] = init['py_ind']
            output['ct_01'] = init['ct_01']
            if 'img_gt_polys' not in output:
                output.update({'img_gt_polys': init['img_gt_polys'] * self.cfg.commen.down_ratio})
            if 'img_gt_init_polys' not in output:
                output.update({'img_gt_init_polys': init['img_gt_init_polys'] * self.cfg.commen.down_ratio})
            if 'img_gt_coarse_polys' not in output:
                output.update({'img_gt_coarse_polys': init['img_gt_coarse_polys'] * self.cfg.commen.down_ratio})
        else:
            output.setdefault('py', []).append(py_48)
            output.setdefault('py_keys', []).append('up_1')
            output['batch_ind'] = output['py_ind']

        for i, gcn in enumerate(self.gcn_stage1):
            return_vtx = (self.cfg.model.use_vertex_classifier and
                          not self.cfg.train.loss_params['vertex_cls']['train_only_final'])
            output = gcn(output, feature_mid, batch,
                           test_stage=self.test_stage, cfg=self.cfg,
                           return_vertex_classifier=return_vtx)
            contour_pre = output['py_pred'][-1] if 'py_pred' in output else output['py'][-1]
            if self.refine_pixel is not None:
                refined_pixelmap, contour_map = self.refine_pixel(
                    contour_pre, feature_mid, batch_ind=output['batch_ind'],
                    pre_pixel_map=output['pixel'][-1] if self.cfg.model.ccp_refine_with_pre_pixel else None
                )
                output['pixel'].append(refined_pixelmap)
                output['contour_map'].append(contour_map)

        # 48 -> 96 (+1/edge) 후, stage96 만큼 반복
        cur = output['py_pred'][-1] if 'py_pred' in output else output['py'][-1]  # (B,48,2)
        py_96 = self._densify_uniform(cur, k_insert_per_edge=1)  # (B,96,2)
        if self.training:
            output.setdefault('py_pred', []).append(py_96)
            output.setdefault('py_keys', []).append('up_2')
        else:
            output.setdefault('py', []).append(py_96)
            output.setdefault('py_keys', []).append('up_2')

        for i, gcn in enumerate(self.gcn_stage2):
            # 보통 마지막 단계에서 vertex classifier를 반환
            is_last = (i == len(self.gcn_stage2) - 1)
            return_vtx = self.cfg.model.use_vertex_classifier and (is_last or
                                                                   not self.cfg.train.loss_params['vertex_cls'][
                                                                       'train_only_final'])
            output = gcn(output, feature_mid, batch,
                           test_stage=self.test_stage, cfg=self.cfg,
                           return_vertex_classifier=return_vtx)
            contour_pre = output['py_pred'][-1] if 'py_pred' in output else output['py'][-1]
            if self.refine_pixel is not None:
                refined_pixelmap, contour_map = self.refine_pixel(
                    contour_pre, feature_mid, batch_ind=output['batch_ind'],
                    pre_pixel_map=output['pixel'][-1] if self.cfg.model.ccp_refine_with_pre_pixel else None
                )
                output['pixel'].append(refined_pixelmap)
                output['contour_map'].append(contour_map)

        # ===== 7) (옵션) 회전 복원 / vertex reduction / simple check =====
        if 'test' in batch['meta'] and self.cfg.test.single_rotate_angle is not None:
            for k in ['poly_init', 'poly_coarse']:
                output[k] = inverse_rotate_contours(
                    output[k], angle=self.cfg.test.single_rotate_angle,
                    height=self.cfg.data.test_scale[0], width=self.cfg.data.test_scale[1]
                )
            for i in range(len(output['py'])):
                output['py'][i] = inverse_rotate_contours(
                    output['py'][i], angle=self.cfg.test.single_rotate_angle,
                    height=self.cfg.data.test_scale[0], width=self.cfg.data.test_scale[1]
                )

        if (not self.training) and self.cfg.model.use_vertex_classifier:
            reduced = self.reduce_vertex(output, min_vertices=self.cfg.test.reduce_min_vertices,
                                         step=self.cfg.test.reduce_step)
            output['py_reduced'] = reduced
            output['is_simple_reduced'] = check_simply_connected(reduced)

        if (not self.training) and self.cfg.test.check_simple and (len(output['py'][-1]) > 0):
            output['is_simple'] = check_simply_connected(output['py'][-1])

        return output


class ContourRNN(nn.Module):
    def __init__(self, cfg=None):
        super(ContourRNN, self).__init__()
        self.cfg = cfg
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.use_gt_det = cfg.train.use_gt_det

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          input_channels=4 if cfg.model.type_add_pixel_mask == 'concat' else 3,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          use_bn_in_head=cfg.model.use_bn_in_head,
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))
        self.train_decoder = Decode(num_point=cfg.commen.init_points_per_poly, init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score, c_in=self.dla.base.channels[self.dla.first_level]+1 if cfg.model.add_grad_feature else self.dla.base.channels[self.dla.first_level],
                                    num_point_each_step=cfg.commen.points_per_poly_steps,
                                    cat_include_coarse=cfg.model.cat_include_coarse)
        self.contour_deform = RefineRNN(hidden_dims=cfg.model.rnn_params['hidden_dims'] if 'hidden_dims' in cfg.model.rnn_params else [64,16],
                                     type=cfg.model.rnn_params['type'] if 'type' in cfg.model.rnn_params else 'dynamic_hidden_dim',
                                     num_layers=cfg.model.rnn_params['num_layers'] if 'num_layers' in cfg.model.rnn_params else 2, num_dec=cfg.model.evolve_iters,
                                     bidirectional=cfg.model.rnn_params['bidirectional'] if 'bidirectional' in cfg.model.rnn_params else False,
                                     fc_dims=cfg.model.rnn_params['fc_dims'] if 'fc_dims' in cfg.model.rnn_params else [2],
                                     layer_type=cfg.model.rnn_params['layer_type'] if 'layer_type' in cfg.model.rnn_params else 'linear',
                                     input_window_size=cfg.model.rnn_params['input_window_size'] if 'input_window_size' in cfg.model.rnn_params else (7,7),
                                     input_window_stride=cfg.model.rnn_params['input_window_stride'] if 'input_window_stride' in cfg.model.rnn_params else 1,
                                     fc_type=cfg.model.rnn_params['fc_type'] if 'fc_type' in cfg.model.rnn_params else 'dim_reduction_first',
                                     fc_activation=cfg.model.rnn_params['fc_activation'] if 'fc_activation' in cfg.model.rnn_params else 'none',
                                     c_in=cfg.model.rnn_params['c_in'] if 'c_in' in cfg.model.rnn_params else self.dla.base.channels[self.dla.first_level],
                                     rnn_type=cfg.model.rnn_params['rnn_type'] if 'rnn_type' in cfg.model.rnn_params else 'GRU',
                                        grad_feature_neighbors=cfg.model.evolve_grad_feature_neighbors, num_points=cfg.commen.points_per_poly,
                                        ro=cfg.model.coarse_stride,
                                        use_input_with_rel_ind=cfg.model.evolve_use_input_with_rel_ind,
                                        is_dec_exclusive=cfg.model.evolve_is_dec_exclusive,
                                        aggregate_feature=cfg.model.rnn_params['aggregate_feature'] if 'aggregate_feature' in cfg.model.rnn_params else False,
                                        refine_kernel_size=cfg.model.refine_kernel_size,
                                        aggregate_type=cfg.model.rnn_params['aggregate_type'] if 'aggregate_type' in cfg.model.rnn_params else 'default',
                                        aggregate_fusion_conv_num=cfg.model.rnn_params['aggregate_fusion_conv_num'] if 'aggregate_fusion_conv_num' in cfg.model.rnn_params else 3,
                                        aggregate_fusion_state_dim=cfg.model.rnn_params['aggregate_fusion_state_dim'] if 'aggregate_fusion_state_dim' in cfg.model.rnn_params else 256,
                                        bi_comb_type=cfg.model.rnn_params['bi_comb_type'] if 'bi_comb_type' in cfg.model.rnn_params else 'none')
        self.net_preprocess()

    def forward(self, input_im, batch=None):
        if self.cfg.model.type_add_pixel_mask == 'concat':
            input = torch.cat([input_im,batch['pixel_gt'].to(input_im.device).unsqueeze(1)], dim=1)
        else:
            input = input_im
        output, cnn_feature, feature_banks = self.dla(input)
        # output, cnn_feature = self.dla(input_im)
        if 'test' not in batch['meta']:
            self.train_decoder(batch, cnn_feature, output, is_training=True)
        else:
            with torch.no_grad():
                self.train_decoder(batch, cnn_feature, output, is_training=False, ignore_gloabal_deform=False)
        # LSTM
        # in : output['poly_coarse'], cnn_feature
        # out : output['py']/output['py_pred'], output['batch_ind']
        output = self.contour_deform(output, cnn_feature, batch)
        if (not self.training) and self.cfg.test.check_simple and (output['py'][-1].shape[0] > 0):
            output['is_simple'] = check_simply_connected(output['py'][-1])
        return output

    def net_preprocess(self):
        fix_net_names = []
        if self.cfg.train.fix_dla:
            fix_net_names.append('dla')
        if self.cfg.train.fix_decode:
            fix_net_names.append('train_decoder')
        if self.cfg.train.fix_deform:
            fix_net_names.append('poly_deform')
        fix_network(self, fix_net_names)

class DeepSnake(nn.Module):
    def __init__(self, cfg=None):
        super(DeepSnake, self).__init__()
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.use_gt_det = cfg.train.use_gt_det
        self.cfg = cfg

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          use_bn_in_head=cfg.model.use_bn_in_head,
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))
        self.stride = cfg.model.init_stride
        self.down_sample = cfg.commen.down_ratio
        self.min_ct_score = cfg.test.ct_score
        self.poly_deform = SnakeEvolution(cfg=self.cfg, evole_ietr_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.commen.down_ratio, with_img_idx=cfg.model.with_img_idx, refine_kernel_size=cfg.model.refine_kernel_size)
        # self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
        #                             coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
        #                             min_ct_score=cfg.test.ct_score)
        # self.poly_deform = DecoderLSTM(hidden_dims=cfg.model.lstm_hidden_dims, type=cfg.model.lstm_hidden_type,
        #                                num_layers=cfg.model.lstm_n_layer, num_dec=cfg.model.evolve_iters,
        #                                ro=cfg.model.coarse_stride, use_input_with_rel_ind=cfg.model.evolve_use_input_with_rel_ind,
        #                                is_dec_exclusive=cfg.model.evolve_is_dec_exclusive,
        #                                bidirectional=cfg.model.lstm_bidirectional, fc_dims=cfg.model.lstm_fc_dims,
        #                                lstm_type=cfg.model.lstm_type, lstm_input_window_size=cfg.model.lstm_input_window_size, lstm_input_window_stride=cfg.model.lstm_input_window_stride,
        #                                lstm_fc_type=cfg.model.lstm_fc_type, lstm_fc_activation=cfg.model.lstm_fc_activation_type,
        #                                grad_feature_neighbors=cfg.model.evolve_grad_feature_neighbors)

    def net_preprocess(self):
        return 0

    def decode_detection(self, output, h, w):
        ct_hm = output['ct_hm']
        wh = output['wh']
        ct, detection = decode_ct_hm_snake(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return output

    def forward(self, input_im, batch=None):
        output, cnn_feature, feature_banks = self.dla(input_im)
        with torch.no_grad():
            output = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))

        output = self.poly_deform(output, cnn_feature, batch)
        if (not self.training) and self.cfg.test.check_simple and (output['py'][-1].shape[0] > 0):
            output['is_simple'] = check_simply_connected(output['py'][-1])
        return output

class PolyLSTMNet(nn.Module):
    def __init__(self, cfg=None):
        super(PolyLSTMNet, self).__init__()
        self.cfg = cfg
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.use_gt_det = cfg.train.use_gt_det

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=cfg.model.dla_pretrained,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          input_channels=4 if cfg.model.type_add_pixel_mask == 'concat' else 3,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))
        self.train_decoder = Decode(num_point=cfg.commen.init_points_per_poly, init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score, c_in=self.dla.base.channels[self.dla.first_level]+1 if cfg.model.add_grad_feature else self.dla.base.channels[self.dla.first_level],
                                    num_point_each_step=cfg.commen.points_per_poly_steps,
                                    cat_include_coarse=cfg.model.cat_include_coarse)
        self.poly_deform = DecoderLSTM(hidden_dims=cfg.model.lstm_hidden_dims, type=cfg.model.lstm_hidden_type,
                                       num_layers=cfg.model.lstm_n_layer, num_dec=cfg.model.evolve_iters,
                                       ro=cfg.model.coarse_stride, use_input_with_rel_ind=cfg.model.evolve_use_input_with_rel_ind,
                                       is_dec_exclusive=cfg.model.evolve_is_dec_exclusive,
                                       bidirectional=cfg.model.lstm_bidirectional, fc_dims=cfg.model.lstm_fc_dims,
                                       lstm_type=cfg.model.lstm_type, lstm_input_window_size=cfg.model.lstm_input_window_size, lstm_input_window_stride=cfg.model.lstm_input_window_stride,
                                       lstm_fc_type=cfg.model.lstm_fc_type, lstm_fc_activation=cfg.model.lstm_fc_activation_type,
                                       grad_feature_neighbors=cfg.model.evolve_grad_feature_neighbors, num_points=cfg.commen.points_per_poly,
                                       c_in=self.dla.base.channels[self.dla.first_level]+1 if cfg.model.add_grad_feature else self.dla.base.channels[self.dla.first_level])
        self.net_preprocess()

    def forward(self, input_im, batch=None):
        if self.cfg.model.type_add_pixel_mask == 'concat':
            input = torch.cat([input_im,batch['pixel_gt'].to(input_im.device).unsqueeze(1)], dim=1)
        else:
            input = input_im
        output, cnn_feature, feature_banks = self.dla(input)
        if 'test' not in batch['meta']:
            self.train_decoder(batch, cnn_feature, output, is_training=True)
        else:
            with torch.no_grad():
                self.train_decoder(batch, cnn_feature, output, is_training=False, ignore_gloabal_deform=False)
        # LSTM
        # in : output['poly_coarse'], cnn_feature
        # out : output['py']/output['py_pred'], output['batch_ind']
        output = self.poly_deform(output, cnn_feature, batch)
        return output

    def net_preprocess(self):
        fix_net_names = []
        if self.cfg.train.fix_dla:
            fix_net_names.append('dla')
        if self.cfg.train.fix_decode:
            fix_net_names.append('train_decoder')
        if self.cfg.train.fix_deform:
            fix_net_names.append('poly_deform')
        fix_network(self, fix_net_names)

class SnakeInitContourNet(nn.Module):
    def __init__(self, cfg=None):
        super(SnakeInitContourNet, self).__init__()
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.use_gt_det = cfg.train.use_gt_det
        self.cfg = cfg

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))
        self.stride = cfg.model.init_stride
        self.down_sample = cfg.commen.down_ratio
        self.min_ct_score = cfg.test.ct_score
        self.poly_deform = SnakeEvolution(use_part='init', cfg=self.cfg, evole_ietr_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.commen.down_ratio, with_img_idx=cfg.model.with_img_idx, refine_kernel_size=cfg.model.refine_kernel_size)
        # self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
        #                             coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
        #                             min_ct_score=cfg.test.ct_score)
        # self.poly_deform = DecoderLSTM(hidden_dims=cfg.model.lstm_hidden_dims, type=cfg.model.lstm_hidden_type,
        #                                num_layers=cfg.model.lstm_n_layer, num_dec=cfg.model.evolve_iters,
        #                                ro=cfg.model.coarse_stride, use_input_with_rel_ind=cfg.model.evolve_use_input_with_rel_ind,
        #                                is_dec_exclusive=cfg.model.evolve_is_dec_exclusive,
        #                                bidirectional=cfg.model.lstm_bidirectional, fc_dims=cfg.model.lstm_fc_dims,
        #                                lstm_type=cfg.model.lstm_type, lstm_input_window_size=cfg.model.lstm_input_window_size, lstm_input_window_stride=cfg.model.lstm_input_window_stride,
        #                                lstm_fc_type=cfg.model.lstm_fc_type, lstm_fc_activation=cfg.model.lstm_fc_activation_type,
        #                                grad_feature_neighbors=cfg.model.evolve_grad_feature_neighbors)

    def net_preprocess(self):
        return 0

    def decode_detection(self, output, h, w):
        ct_hm = output['ct_hm']
        wh = output['wh']
        ct, detection = decode_ct_hm_snake(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return output

    def forward(self, input_im, batch=None):
        output, cnn_feature, feature_banks = self.dla(input_im)
        with torch.no_grad():
            output = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))

        output = self.poly_deform(output, cnn_feature, batch)
        return output

class SnakeCoarseContourNet(nn.Module):
    def __init__(self, cfg=None):
        super(SnakeCoarseContourNet, self).__init__()
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.use_gt_det = cfg.train.use_gt_det
        self.cfg = cfg

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))
        self.stride = cfg.model.init_stride
        self.down_sample = cfg.commen.down_ratio
        self.min_ct_score = cfg.test.ct_score
        self.poly_deform = SnakeEvolution(use_part='coarse', cfg=self.cfg, evole_ietr_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.commen.down_ratio, with_img_idx=cfg.model.with_img_idx, refine_kernel_size=cfg.model.refine_kernel_size)
        # self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
        #                             coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
        #                             min_ct_score=cfg.test.ct_score)
        # self.poly_deform = DecoderLSTM(hidden_dims=cfg.model.lstm_hidden_dims, type=cfg.model.lstm_hidden_type,
        #                                num_layers=cfg.model.lstm_n_layer, num_dec=cfg.model.evolve_iters,
        #                                ro=cfg.model.coarse_stride, use_input_with_rel_ind=cfg.model.evolve_use_input_with_rel_ind,
        #                                is_dec_exclusive=cfg.model.evolve_is_dec_exclusive,
        #                                bidirectional=cfg.model.lstm_bidirectional, fc_dims=cfg.model.lstm_fc_dims,
        #                                lstm_type=cfg.model.lstm_type, lstm_input_window_size=cfg.model.lstm_input_window_size, lstm_input_window_stride=cfg.model.lstm_input_window_stride,
        #                                lstm_fc_type=cfg.model.lstm_fc_type, lstm_fc_activation=cfg.model.lstm_fc_activation_type,
        #                                grad_feature_neighbors=cfg.model.evolve_grad_feature_neighbors)

    def net_preprocess(self):
        return 0

    def decode_detection(self, output, h, w):
        ct_hm = output['ct_hm']
        wh = output['wh']
        ct, detection = decode_ct_hm_snake(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return output

    def forward(self, input_im, batch=None):
        output, cnn_feature, feature_banks = self.dla(input_im)
        with torch.no_grad():
            output = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))

        output = self.poly_deform(output, cnn_feature, batch)
        return output

class InitContourNet(nn.Module):
    def __init__(self, cfg=None):
        super(InitContourNet, self).__init__()
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.use_gt_det = cfg.train.use_gt_det
        self.cfg = cfg

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))
        self.stride = cfg.model.init_stride
        self.down_sample = cfg.commen.down_ratio
        self.min_ct_score = cfg.test.ct_score
        # self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
        #                             coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
        #                             min_ct_score=cfg.test.ct_score)
        # self.poly_deform = DecoderLSTM(hidden_dims=cfg.model.lstm_hidden_dims, type=cfg.model.lstm_hidden_type,
        #                                num_layers=cfg.model.lstm_n_layer, num_dec=cfg.model.evolve_iters,
        #                                ro=cfg.model.coarse_stride, use_input_with_rel_ind=cfg.model.evolve_use_input_with_rel_ind,
        #                                is_dec_exclusive=cfg.model.evolve_is_dec_exclusive,
        #                                bidirectional=cfg.model.lstm_bidirectional, fc_dims=cfg.model.lstm_fc_dims,
        #                                lstm_type=cfg.model.lstm_type, lstm_input_window_size=cfg.model.lstm_input_window_size, lstm_input_window_stride=cfg.model.lstm_input_window_stride,
        #                                lstm_fc_type=cfg.model.lstm_fc_type, lstm_fc_activation=cfg.model.lstm_fc_activation_type,
        #                                grad_feature_neighbors=cfg.model.evolve_grad_feature_neighbors)

    def net_preprocess(self):
        return 0

    def forward(self, input_im, batch=None):
        output, cnn_feature, feature_banks = self.dla(input_im)
        # if 'test' not in batch['meta']:
        #     self.train_decoder(batch, cnn_feature, output, is_training=True)
        # else:
        #     with torch.no_grad():
        #         self.train_decoder(batch, cnn_feature, output, is_training=False, ignore_gloabal_deform=False)
        # LSTM
        # in : output['poly_coarse'], cnn_feature
        # out : output['py']/output['py_pred'], output['batch_ind']
        # output = self.poly_deform(output, cnn_feature, batch)
        if self.training:
            wh_pred = output['wh']
            # print(f"wh_pred : {wh_pred.shape}")
            ct_01 = batch['ct_01'].bool()
            ct_ind = batch['ct_ind'][ct_01]
            ct_img_idx = batch['ct_img_idx'][ct_01]
            _, _, height, width = batch['ct_hm'].size()
            ct_x = ct_ind % width
            ct_y = torch.div(ct_ind, width, rounding_mode='trunc')

            # print(f"wh_pred[ct_img_idx, :, ct_y, ct_x] : {wh_pred[ct_img_idx, :, ct_y, ct_x].shape}")

            if ct_x.size(0) == 0:
                ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), 1, 2)
            else:
                ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), -1, 2)

            # print(f"ct_offset : {ct_offset.shape}")

            ct_x, ct_y = ct_x[:, None].to(torch.float32), ct_y[:, None].to(torch.float32)
            ct = torch.cat([ct_x, ct_y], dim=1)

            init_polys = ct_offset * self.stride + ct.unsqueeze(1).expand(ct_offset.size(0),
                                                                          ct_offset.size(1), ct_offset.size(2))
            # coarse_polys = self.refine(cnn_feature, ct, init_polys, ct_img_idx.clone())

            output.update({'poly_init': init_polys * self.down_sample})
            # output.update({'poly_coarse': coarse_polys * self.down_sample})
            # ct_01 = batch['ct_01'].byte()
            ct_01 = batch['ct_01'].bool()
            output.update({'img_gt_polys': collect_training(batch['img_gt_polys'], ct_01)})
            output.update({'img_gt_init_polys': collect_training(batch['img_gt_init_polys'], ct_01)})
            # init.update({'img_init_polys': ret['poly_coarse'].detach() / ro})
            # can_init_polys = img_poly_to_can_poly(ret['poly_coarse'].detach() / ro)
            # init.update({'can_init_polys': can_init_polys})

            ct_num = batch['meta']['ct_num']
            output.update({'batch_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
            # init.update({'py_ind': init['py_ind']})
            output['batch_ind'] = output['batch_ind'].to(init_polys.device)
            output['img_gt_polys'] = output['img_gt_polys'] * self.down_sample
            output['img_gt_init_polys'] = output['img_gt_init_polys'] * self.down_sample
        else:
            K = 100
            hm_pred, wh_pred = output['ct_hm'], output['wh']
            poly_init, detection, wh_pred = decode_ct_hm(torch.sigmoid(hm_pred), wh_pred,
                                                         K=K, stride=self.stride)
            valid = detection[0, :, 2] >= self.min_ct_score
            poly_init, detection, wh_pred = poly_init[0][valid], detection[0][valid], wh_pred[0][valid]

            init_polys = clip_to_image(poly_init, cnn_feature.size(2), cnn_feature.size(3))
            output.update({'poly_init': init_polys * self.down_sample})

            # img_id = torch.zeros((len(poly_init),), dtype=torch.int64)
            # poly_coarse = self.refine(cnn_feature, detection[:, :2], poly_init, img_id, ignore=ignore_gloabal_deform)
            # coarse_polys = clip_to_image(poly_coarse, cnn_feature.size(2), cnn_feature.size(3))
            # output.update({'poly_coarse': coarse_polys * self.down_sample})
            output.update({'detection': detection, 'wh_pred': wh_pred})
            output['py'] = [output['poly_init']]
            output['batch_ind'] = torch.zeros((init_polys.size(0), ), dtype=torch.int32, device=init_polys.device)
        return output

class CoarseContourNet(nn.Module):
    def __init__(self, cfg=None):
        super(CoarseContourNet, self).__init__()
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.use_gt_det = cfg.train.use_gt_det
        self.cfg = cfg

        pre_defined_bn_dla = self._get_pre_defined_bn(cfg.train.load_trained_rasterizer, filters=['dla', 'layer.1.'])
        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))
        self.stride = cfg.model.init_stride
        self.down_sample = cfg.commen.down_ratio
        self.min_ct_score = cfg.test.ct_score
        self.train_decoder = Decode(num_point=cfg.commen.init_points_per_poly, init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score, c_in=self.dla.base.channels[self.dla.first_level]+1 if cfg.model.add_grad_feature else self.dla.base.channels[self.dla.first_level],
                                    num_point_each_step=cfg.commen.points_per_poly_steps,
                                    with_img_idx=cfg.model.with_img_idx, refine_kernel_size=cfg.model.refine_kernel_size,
                                    cat_include_coarse=cfg.model.cat_include_coarse)
        # self.poly_deform = DecoderLSTM(hidden_dims=cfg.model.lstm_hidden_dims, type=cfg.model.lstm_hidden_type,
        #                                num_layers=cfg.model.lstm_n_layer, num_dec=cfg.model.evolve_iters,
        #                                ro=cfg.model.coarse_stride, use_input_with_rel_ind=cfg.model.evolve_use_input_with_rel_ind,
        #                                is_dec_exclusive=cfg.model.evolve_is_dec_exclusive,
        #                                bidirectional=cfg.model.lstm_bidirectional, fc_dims=cfg.model.lstm_fc_dims,
        #                                lstm_type=cfg.model.lstm_type, lstm_input_window_size=cfg.model.lstm_input_window_size, lstm_input_window_stride=cfg.model.lstm_input_window_stride,
        #                                lstm_fc_type=cfg.model.lstm_fc_type, lstm_fc_activation=cfg.model.lstm_fc_activation_type,
        #                                grad_feature_neighbors=cfg.model.evolve_grad_feature_neighbors,
        #                                c_in=65 if cfg.model.add_grad_feature else 64)
        self.gaussian = torchvision.transforms.GaussianBlur(6*cfg.model.grad_feature_params['sigma']+1, sigma=cfg.model.grad_feature_params['sigma'])
        if self.cfg.model.with_rasterize_net:
            pre_defined_bn = self._get_pre_defined_bn(cfg.train.load_trained_rasterizer, filters=['rasterizer','layer.1.'])
            raster_out_size = [int(self.cfg.data.input_w / self.cfg.commen.down_ratio),
                               int(self.cfg.data.input_h / self.cfg.commen.down_ratio)] if self.cfg.model.is_raster_down_sampled else [self.cfg.data.input_w, self.cfg.data.input_h]
            self.rasterizer = Rasterizer(rasterize_type=self.cfg.model.raster_type,
                                         sigma=self.cfg.model.raster_sigma,
                                         out_size=raster_out_size,
                                         pre_defined_bn=pre_defined_bn,
                                         scale=self.cfg.model.raster_scale,
                                         **self.cfg.model.raster_netparams)
        else:
            self.rasterizer = None

    def _get_pre_defined_bn(self, model_path, filters=[]):
        if os.path.exists(model_path):
            pretrained_model = torch.load(model_path)['net']
            pre_defined_bn = {}
            for key in pretrained_model.keys():
                filters_if = [f in key for f in filters]
                if all(filters_if) and ('num_batches_tracked' not in key):
                    layer_name = key.split('.')[2]
                    param_name = key.split('.')[-1]
                    if layer_name in pre_defined_bn:
                        pre_defined_bn[layer_name].update({param_name: pretrained_model[key]})
                    else:
                        pre_defined_bn[layer_name] = {param_name: pretrained_model[key]}
            return pre_defined_bn
        else:
            print(f"Could not load pre-defined BN : {model_path}")
            return None

    def net_preprocess(self):
        fix_net_names = []
        if self.cfg.train.fix_dla:
            fix_net_names.append('dla')
        if self.cfg.train.fix_decode:
            fix_net_names.append('train_decoder')
        fix_network(self, fix_net_names)

    def forward(self, input_im, batch=None):
        if self.cfg.train.fix_dla:
            self.dla.eval()
            with torch.no_grad():
                output, cnn_feature, feature_banks = self.dla(input_im)
        else:
            output, cnn_feature, feature_banks = self.dla(input_im)
        if self.cfg.model.add_grad_feature:
            input_im_gray = torch.mean(input_im, 1, keepdim=True)
            feature_gaussian = self.gaussian(input_im_gray)
            weights = torch.tensor([[0., -1., 0.],
                                    [-1., 4., -1.],
                                    [0., -1., 0.]])
            weights = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1).type(feature_gaussian.dtype).to(feature_gaussian.device)
            feature_log = F.conv2d(feature_gaussian, weights, padding='same')
            feature_log = F.max_pool2d(feature_log, 4)
            if 'normalize' in self.cfg.model.grad_feature_params:
                if self.cfg.model.grad_feature_params['normalize'] == 'minmax':
                    feature_log = (feature_log - feature_log.min())/(feature_log.max() - feature_log.min())
                elif self.cfg.model.grad_feature_params['normalize'] == 'max':
                    feature_log = feature_log / feature_log.max()
                elif self.cfg.model.grad_feature_params['normalize'] == 'mean0':
                    feature_log = feature_log - feature_log.mean()
                elif self.cfg.model.grad_feature_params['normalize'] == 'mean0var1':
                    feature_log = (feature_log - feature_log.mean())/feature_log.std()
            # print(f"feature_log : {feature_log.shape}")
            cnn_feature = torch.cat([cnn_feature, feature_log], 1)
        if self.training:
            self.train_decoder(batch, cnn_feature, output, is_training=True)
        else:
            if self.cfg.test.get_featuremap:
                output['fm'] = {'F_backbone': cnn_feature.clone().detach()}
            with torch.no_grad():
                feature_coarse = self.train_decoder(batch, cnn_feature, output, is_training=False, ignore_gloabal_deform=False, get_feature=self.cfg.test.get_featuremap)
            if self.cfg.test.get_featuremap:
                if feature_coarse is None:
                    print("none")
                output['fm'].update({'F_coarse': feature_coarse.clone().detach()})
        # LSTM
        # in : output['poly_coarse'], cnn_feature
        # out : output['py']/output['py_pred'], output['batch_ind']
        # output = self.poly_deform(output, cnn_feature, batch)
        if self.training:
            # ct_01 = batch['ct_01'].byte()
            ct_01 = batch['ct_01'].bool()
            output.update({'img_gt_polys': collect_training(batch['img_gt_polys'], ct_01)})
            output.update({'img_gt_init_polys': collect_training(batch['img_gt_init_polys'], ct_01)})
            output.update({'img_gt_coarse_polys': collect_training(batch['img_gt_coarse_polys'], ct_01)})
            # init.update({'img_init_polys': ret['poly_coarse'].detach() / ro})
            # can_init_polys = img_poly_to_can_poly(ret['poly_coarse'].detach() / ro)
            # init.update({'can_init_polys': can_init_polys})

            ct_num = batch['meta']['ct_num']
            output.update({'batch_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
            # init.update({'py_ind': init['py_ind']})
            output['batch_ind'] = output['batch_ind'].to(output['poly_coarse'].device)
            output['img_gt_polys'] = output['img_gt_polys'] * self.down_sample
            output['img_gt_init_polys'] = output['img_gt_init_polys'] * self.down_sample
            output['img_gt_coarse_polys'] = output['img_gt_coarse_polys'] * self.down_sample
            if self.rasterizer is not None:
                output['pred_mask'] = {'init': self.rasterizer(output['poly_init']/self.down_sample if self.cfg.model.is_raster_down_sampled else output['poly_init']),
                                       'coarse': self.rasterizer(output['poly_coarse']/self.down_sample if self.cfg.model.is_raster_down_sampled else output['poly_coarse'])}
        else:
            if self.cfg.model.with_img_idx:
                if output['ct_01'].size(1) > 0:
                    poly_coarse_reshape = output['poly_coarse'][output['ct_01']]
                else:
                    poly_coarse_reshape = output['poly_coarse']
            else:
                poly_coarse_reshape = output['poly_coarse']
            output['py'] = [output['poly_init'], poly_coarse_reshape]
            if self.cfg.test.check_simple and (output['py'][-1].shape[0] > 0):
                output['is_simple'] = check_simply_connected(output['py'][-1])
            output['batch_ind'] = torch.zeros((output['poly_coarse'].size(0), ), dtype=torch.int32, device=output['poly_coarse'].device)
            if self.rasterizer is not None:
                output['pred_mask'] = {'init': self.rasterizer(output['poly_init']/self.down_sample if self.cfg.model.is_raster_down_sampled else output['poly_init']),
                                       'coarse': self.rasterizer(poly_coarse_reshape/self.down_sample if self.cfg.model.is_raster_down_sampled else poly_coarse_reshape)}
        return output

class RasterizeNet(nn.Module):
    def __init__(self, cfg):
        super(RasterizeNet, self).__init__()
        self.cfg = cfg
        raster_out_size = [int(self.cfg.data.input_w / self.cfg.commen.down_ratio),
                           int(self.cfg.data.input_h / self.cfg.commen.down_ratio)] if self.cfg.model.is_raster_down_sampled else [
            self.cfg.data.input_w, self.cfg.data.input_h]

        self.rasterizer = Rasterizer(rasterize_type=self.cfg.model.raster_type,
                                     sigma=self.cfg.model.raster_sigma,
                                     out_size=raster_out_size,
                                     scale=self.cfg.model.raster_scale,
                                     **self.cfg.model.raster_netparams)

    def net_preprocess(self):
        return 0

    def forward(self, input_contour):
        return {'mask' : self.rasterizer(input_contour)}


class Pixelnet(nn.Module):
    def __init__(self, cfg=None):
        super(Pixelnet, self).__init__()
        self.cfg = cfg
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.model.down_ratio
        heads = cfg.model.heads
        self.use_gt_det = cfg.train.use_gt_det
        self.test_stage = cfg.test.test_stage

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn,
                          input_channels=4 if cfg.model.type_add_pixel_mask == 'concat' else 3,
                          clone_to_ida_up=cfg.model.dla_clone_to_ida_up,
                          concat_upper_layer=cfg.model.concat_upper_layer,
                          use_bn_in_head=cfg.model.use_bn_in_head,
                          head_conv_config=getattr(cfg.model, 'head_conv_config', None),
                          concat_multi_layers=getattr(cfg.model, 'concat_multi_layers', None))

        self.net_preprocess()

    def forward(self, x, batch=None):
        if self.cfg.model.type_add_pixel_mask == 'concat':
            input = torch.cat([x,batch['pixel_gt'].to(x.device).unsqueeze(1)], dim=1)
        else:
            input = x
        output, cnn_feature, feature_banks = self.dla(input)
        output['feature_banks'] = feature_banks
        output['cnn_feature'] = cnn_feature
        feature_deform = cnn_feature

        if 'test' in batch['meta']:
            with torch.no_grad():
                if self.cfg.test.get_featuremap:
                    output['fm'] = {}
                    for k in output['feature_banks'].keys():
                        if isinstance(output['feature_banks'][k], list):
                            for banks_i in range(len(output['feature_banks'][k])):
                                output['fm'].update({f'F_{k}{banks_i}': output['feature_banks'][k][banks_i].clone().detach()})
                        else:
                            output['fm'].update({f'F_{k}': output['feature_banks'][k].clone().detach()})

                    output['fm'].update({'F_backbone': feature_deform.clone().detach()})

        return output

    def net_preprocess(self):
        fix_net_names = []
        if self.cfg.train.fix_dla:
            fix_net_names.append('dla')
        if self.cfg.train.fix_decode:
            fix_net_names.append('train_decoder')
        if self.cfg.train.fix_deform:
            fix_net_names.append('gcn')
        fix_network(self, fix_net_names)


# def load_network(net, model_dir, strict=True):
#     if not os.path.exists(model_dir):
#         print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
#         return 0
#
#     print('load model: {}'.format(model_dir))
#     pretrained_model = torch.load(model_dir)
#     if 'epoch' in pretrained_model.keys():
#         epoch = pretrained_model['epoch'] + 1
#     else:
#         epoch = 0
#     pretrained_model = pretrained_model['net']
#
#     net_weight = net.state_dict()
#     for key in net_weight.keys():
#         net_weight.update({key: pretrained_model[key]})
#
#     net.load_state_dict(net_weight, strict=strict)


def fix_network(net, fix_net_names):
    for subnet in fix_net_names:
        print(f"{subnet} has been frozen.")
        for module in getattr(net,subnet).modules():
            for name, param in module.named_parameters():
                param.requires_grad = False
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

def get_network(cfg):
    if cfg.commen.task.split('+')[0] == 'init':
        network = InitContourNet(cfg)
    elif cfg.commen.task.split('+')[0] == 'coarse':
        network = CoarseContourNet(cfg)
    elif cfg.commen.task.split('+')[0] == 'e2ec':
        network = E2ECnet(cfg)
    elif cfg.commen.task.split('+')[0] == 'ccp':
        network = CCPnet(cfg)
    elif cfg.commen.task.split('+')[0] == 'ccp_maskinit':
        network = CCPnetMaskInit(cfg)
    elif cfg.commen.task.split('+')[0] == 'ccp_pyramid':
        network = CCPnetPyramid(cfg)
    elif cfg.commen.task.split('+')[0] == 'snake_init':
        network = SnakeInitContourNet(cfg)
    elif cfg.commen.task.split('+')[0] == 'snake_coarse':
        network = SnakeCoarseContourNet(cfg)
    elif cfg.commen.task.split('+')[0] == 'deepsnake':
        network = DeepSnake(cfg)
    elif cfg.commen.task.split('+')[0] == 'rnn':
        network = ContourRNN(cfg)
    elif cfg.commen.task.split('+')[0] == 'pixel':
        network = Pixelnet(cfg)
    else:
        network = PolyLSTMNet(cfg)

    if cfg.model.norm_type == 'group':
        if hasattr(network, 'dla'):
            convert_bn_to_gn(network.dla, cfg.model.norm_num_groups)
    return network
