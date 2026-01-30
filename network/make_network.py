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
from .evolve.evolve_ccp import Evolution as EvolutionCCP, RefinePixel
from .detector_decode.utils import decode_ct_hm, clip_to_image
from .evolve.utils import collect_training
from .data_utils import check_simply_connected, has_self_intersection
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
        
        if self.use_pixel_on_init:
            # Step 1: backbone feature 추출 및 pixel head 실행
            output, cnn_feature, feature_banks = self.dla(input, save_banks=False)
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
            output, cnn_feature, feature_banks = self.dla(input, save_banks=False)

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
                                                    get_feature=False)

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


def fix_network(net, fix_net_names):
    for subnet in fix_net_names:
        print(f"{subnet} has been frozen.")
        for module in getattr(net,subnet).modules():
            for name, param in module.named_parameters():
                param.requires_grad = False
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

def get_network(cfg):
    if cfg.commen.task.split('+')[0] == 'ccp':
        network = CCPnet(cfg)
    else:
        raise ValueError(f"Unsupported task: {cfg.commen.task}. Only 'ccp' is supported.")

    if cfg.model.norm_type == 'group':
        if hasattr(network, 'dla'):
            convert_bn_to_gn(network.dla, cfg.model.norm_num_groups)
    return network
