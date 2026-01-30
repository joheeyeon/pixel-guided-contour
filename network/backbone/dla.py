from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

DICT_dla_module = {'base': 'base', 'dlaup': 'dla_up', 'idaup': 'ida_up'}
_ALIGNED_MODES = {"linear", "bilinear", "bicubic", "trilinear"}

def _align_kw(mode: str, align_corners: bool = False):
    """interpolate용 kwargs: 지원 모드일 때만 align_corners를 넣는다."""
    return {"align_corners": align_corners} if mode in _ALIGNED_MODES else {}


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        # f"conv1) x : {x.shape} / residual: {residual.shape}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # f"conv2) x : {out.shape} / residual: {residual.shape}")
        out = self.conv2(out)
        out = self.bn2(out)
        # f"+) x : {out.shape} / residual: {residual.shape}")

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        # f"x : {x.shape} / residual : {residual.shape}")
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False, input_channels=3):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.base_layer = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        if self.input_channels != 3:
            for key, value in model_weights.copy().items():
                if 'base_layer.0.' in key:
                    model_weights.pop(key)
        self.load_state_dict(model_weights, strict=False)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho, use_dcn=True, strid=1):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        if use_dcn:
            from .dcn_v2 import DCN
            self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=strid, padding=1, dilation=1, deformable_groups=1)
            #from mmcv.ops import ModulatedDeformConv2dPack as DCN
            #self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deform_groups=1)
        else:
            self.conv = nn.Conv2d(chi, cho, kernel_size=(3, 3), stride=strid, padding=1, dilation=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f, use_dcn=True):
        super(IDAUp, self).__init__()
        self.channels = channels
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o, use_dcn=use_dcn)
            node = DeformConv(o, o, use_dcn=use_dcn)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None, use_dcn=True):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = list(channels)  # 복사본 생성
        self.channels = list(channels)  # 원본 보존을 위해 복사본 저장
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j], use_dcn=use_dcn))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, **_align_kw(self.mode, align_corners=False)) #edit self.mode -> nearest (for reproduc)
        return x


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, use_dcn=True, input_channels=3, clone_to_ida_up=True,
                 concat_upper_layer=None, use_bn_in_head=True, interpolate_mode='bilinear',
                 head_stage_map=None,  # edit:ccp+ive-fm:25-08-09: {'pixel':'mid', 'ct_hm':'low', 'wh':'low', 'reg':'low'} 등
                 head_conv_config=None,  # 새로운 head_conv 설정 시스템
                 concat_multi_layers=None,  # 새로운 multi-scale concat 옵션: ['base_2', 'base_3'] 등
                 ):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.interpolate_mode = interpolate_mode
        self.use_bn_in_head = use_bn_in_head
        self.clone_to_ida_up = clone_to_ida_up
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained, input_channels=input_channels)
        self.head_stage_map = head_stage_map
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]

        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales, use_dcn=use_dcn)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)], use_dcn=use_dcn)

        self.heads = heads
        self.concat_upper_layer = concat_upper_layer
        self.concat_multi_layers = concat_multi_layers

        # ===== concat_upper_layer로 추가되는 채널 계산 (없으면 0) (edit:debug:by-ccp+ive-concat_upper_layer:25-08-09) =====
        concat_extra_ch = 0
        if self.concat_upper_layer is not None:
            _name, _idx = self.concat_upper_layer.split('_')[0], int(self.concat_upper_layer.split('_')[-1])
            concat_extra_ch = getattr(self, DICT_dla_module[_name]).channels[_idx]
        
        # ===== concat_multi_layers로 추가되는 채널 계산 =====
        concat_multi_extra_ch = 0
        if self.concat_multi_layers is not None:
            for layer_name in self.concat_multi_layers:
                try:
                    _name, _idx = layer_name.split('_')[0], int(layer_name.split('_')[-1])
                    
                    # Index 범위 검증
                    if _name == 'base':
                        max_idx = len(channels) - 1  # 0-5
                    elif _name in ['dlaup', 'idaup']:
                        # DLA up과 IDA up의 실제 channels 길이 사용
                        actual_channels = getattr(self, DICT_dla_module[_name]).channels
                        max_idx = len(actual_channels) - 1  # 실제 채널 배열 길이 기준
                        # 실제 채널 확인됨
                    else:
                        raise ValueError(f"Unknown module name: {_name}")
                    
                    if _idx < 0 or _idx > max_idx:
                        raise ValueError(f"Index {_idx} out of range [0, {max_idx}] for {_name}")
                    
                    module_obj = getattr(self, DICT_dla_module[_name])
                    ch_val = module_obj.channels[_idx]
                    concat_multi_extra_ch += ch_val
                    # concat_multi_layers 채널 추가
                except Exception as e:
                    print(f"[DLA INIT ERROR] Exception for '{layer_name}': {e}")
                    print(f"[DLA INIT ERROR] _name: {_name}, _idx: {_idx}")
                    print(f"[DLA INIT ERROR] DICT_dla_module mapping: {DICT_dla_module}")
                    import traceback
                    traceback.print_exc()
                    raise ValueError(f"Invalid concat_multi_layers entry '{layer_name}': {e}")

        # ===== IDAUp 이후 출력 채널 (y[...]의 채널 수) (edit:debug:by-ccp+ive-concat_upper_layer:25-08-09) =====
        ida_out_ch = out_channel if out_channel > 0 else channels[self.first_level]

        # >>> 여기! head in_channels 는 (IDA 채널 + concat 채널) 이어야 함 (edit:debug:by-ccp+ive-concat_upper_layer:25-08-09)
        # 기존 concat_upper_layer와 새로운 concat_multi_layers 모두 고려
        total_concat_ch = concat_extra_ch + concat_multi_extra_ch
        in_ch_mid = ida_out_ch + total_concat_ch
        in_ch_low = ida_out_ch + total_concat_ch
        
        # 디버그 출력 (제거됨)
        

        # ===== edit:ccp+ive-fm:25-08-09: head → stage(mid/low) 맵핑 저장 (기본: 모두 mid) =====
        # - None: 100% 기존 동작 (모든 head가 mid에 붙음)
        # - dict 주면 해당 head는 low or mid 중 선택
        self._head_stage_map = {}
        if head_stage_map is not None:
            for k in heads.keys():
                v = head_stage_map.get(k, 'mid')
                self._head_stage_map[k] = 'low' if v == 'low' else 'mid'
        else:
            for k in heads.keys():
                self._head_stage_map[k] = 'mid'

        # stage 분리 여부 판단 (low를 쓰는 head가 1개라도 있으면 True)
        self.stage_split = any(self._head_stage_map.get(h, 'mid') == 'low' for h in self.heads)

        # ===== low_from_mid (옵션): low가 필요할 때만 만들기 =====
        if self.stage_split:
            self.low_from_mid_stride = 2  # ← 낮추고 싶으면 2
            low_in = in_ch_mid  # y[...]는 모두 out_channel 채널
            low_out = in_ch_low
            self.low_from_mid = DeformConv(low_in, low_out, use_dcn=use_dcn, strid=self.low_from_mid_stride)

        # ===== 공통 빌더 =====
        def _build_head(classes, head, head_conv_cfg, pre_feat, final_kernel):
            # head_conv_config 지원하는 새로운 빌드 로직
            conv_config = head_conv_config or {}
            
            # head별 개별 설정 확인
            if isinstance(conv_config, dict) and head in conv_config:
                # 새로운 head_conv_config 시스템 사용
                head_specific = conv_config[head]
                kernel_sizes = head_specific.get('kernel_sizes', [3, final_kernel])
                channels = head_specific.get('channels', None)
                use_relu = head_specific.get('use_relu', [True])
                padding_mode = head_specific.get('padding', 'auto')
                
                # channels 설정: None이면 기본값 사용
                if channels is None:
                    if isinstance(head_conv_cfg, int) and head_conv_cfg > 0:
                        channels = [head_conv_cfg, classes]
                    else:
                        channels = [classes]
                else:
                    channels = channels + [classes]  # 마지막에 classes 추가
                
                # use_relu 리스트 길이 조정
                if len(use_relu) < len(kernel_sizes) - 1:
                    use_relu = use_relu + [True] * (len(kernel_sizes) - 1 - len(use_relu))
                
                # 레이어 구성
                if len(channels) > 1:
                    layers = []
                    c_current = pre_feat
                    
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
                    # 단일 layer
                    kernel_size = kernel_sizes[-1] if kernel_sizes else final_kernel
                    padding = (kernel_size - 1) // 2 if padding_mode == 'auto' else padding_mode
                    fc = nn.Conv2d(pre_feat, classes, kernel_size=kernel_size,
                                  stride=1, padding=padding, bias=True)
                
                # bias 초기화
                if 'hm' in head:
                    if isinstance(fc, nn.Sequential):
                        fc[-1].bias.data.fill_(-2.19)
                    else:
                        fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
                
                return fc
            
            # 기존 로직 (legacy 지원)
            elif isinstance(head_conv_cfg, int):
                if head_conv_cfg > 0:
                    fc = nn.Sequential(
                        nn.Conv2d(pre_feat, head_conv_cfg, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv_cfg, classes, kernel_size=final_kernel, stride=1,
                                  padding=final_kernel // 2, bias=True)
                    )
                    if 'hm' in head:
                        fc[-1].bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)
                else:
                    fc = nn.Conv2d(pre_feat, classes, kernel_size=final_kernel, stride=1,
                                   padding=final_kernel // 2, bias=True)
                    if 'hm' in head:
                        fc.bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)
                return fc

            elif isinstance(head_conv_cfg, list):
                fc = nn.Sequential()
                cur_feat = pre_feat
                for i_feat, n_feat in enumerate(head_conv_cfg):
                    fc.add_module(f"conv_{i_feat}",
                                  nn.Conv2d(cur_feat, n_feat, kernel_size=3, padding=1, bias=True))
                    fc.add_module(f"relu_{i_feat}", nn.ReLU(inplace=True))
                    cur_feat = n_feat
                fc.add_module("conv_out", nn.Conv2d(cur_feat, classes, kernel_size=final_kernel, stride=1,
                                                    padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
                return fc

            else:
                # dict-like (per-head 구조)
                fc = nn.Sequential()
                cur_feat = pre_feat
                hc = head_conv_cfg[head]
                if isinstance(hc, int):
                    if hc > 0:
                        fc = nn.Sequential(
                            nn.Conv2d(cur_feat, hc, kernel_size=3, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hc, classes, kernel_size=final_kernel, stride=1,
                                      padding=final_kernel // 2, bias=True)
                        )
                        if 'hm' in head:
                            fc[-1].bias.data.fill_(-2.19)
                        else:
                            fill_fc_weights(fc)
                    else:
                        fc = nn.Conv2d(cur_feat, classes, kernel_size=final_kernel, stride=1,
                                       padding=final_kernel // 2, bias=True)
                        if 'hm' in head:
                            fc.bias.data.fill_(-2.19)
                        else:
                            fill_fc_weights(fc)
                    return fc

                elif isinstance(hc, list):
                    for i_feat, n_feat in enumerate(hc):
                        fc.add_module(f"conv_{i_feat}",
                                      nn.Conv2d(cur_feat, n_feat, kernel_size=3, padding=1, bias=True))
                        fc.add_module(f"relu_{i_feat}", nn.ReLU(inplace=True))
                        cur_feat = n_feat
                    fc.add_module("conv_out", nn.Conv2d(cur_feat, classes, kernel_size=final_kernel, stride=1,
                                                        padding=final_kernel // 2, bias=True))
                    if 'hm' in head:
                        fc[-1].bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)
                    return fc

                else:
                    # module_structure dict (conv_*, up_*, convtranspose_*)
                    for layer in hc.keys():
                        if layer.split('_')[0] == 'conv':
                            for convi in range(len(hc[layer]) if isinstance(hc[layer], list) else 1):
                                n_feat = hc[layer][convi] if isinstance(hc[layer], list) else hc[layer]
                                fc.add_module(f"conv_{layer.split('_')[-1]}_{convi}",
                                              nn.Conv2d(cur_feat, n_feat, kernel_size=3, padding=1, bias=True))
                                if self.use_bn_in_head:
                                    fc.add_module(f"bn_{layer.split('_')[-1]}_{convi}",
                                                  nn.BatchNorm2d(n_feat, momentum=BN_MOMENTUM))
                                fc.add_module(f"relu_{layer.split('_')[-1]}_{convi}", nn.ReLU(inplace=True))
                                cur_feat = n_feat
                        elif layer.split('_')[0] == 'up':
                            fc.add_module(f"up_{layer.split('_')[-1]}",
                                          nn.Upsample(scale_factor=hc[layer], mode=layer.split('_')[1]))
                        else:  # convtranspose
                            n_feat = hc[layer][-1]
                            fc.add_module(f"convtranspose_{layer.split('_')[-1]}",
                                          nn.ConvTranspose2d(cur_feat, n_feat, kernel_size=hc[layer][0],
                                                             stride=hc[layer][0]))
                            if self.use_bn_in_head:
                                fc.add_module(f"bntranspose_{layer.split('_')[-1]}",
                                              nn.BatchNorm2d(n_feat, momentum=BN_MOMENTUM))
                            fc.add_module(f"relutranspose_{layer.split('_')[-1]}", nn.ReLU(inplace=True))
                            cur_feat = n_feat

                    fc.add_module("conv_out", nn.Conv2d(cur_feat, classes, kernel_size=final_kernel, stride=1,
                                                        padding=final_kernel // 2, bias=True))
                    fill_fc_weights(fc)
                    return fc

        # ===== per-head로 stage를 선택하여 모듈 생성 =====
        for head in self.heads:
            classes = self.heads[head]
            # mid/low 선택
            stage = self._head_stage_map.get(head, 'mid')  # 'mid' or 'low'
            pre_feat = in_ch_mid if stage == 'mid' else in_ch_low
            
            # 디버그 출력
            print(f"[DLA HEAD BUILD] {head}: stage={stage}, pre_feat={pre_feat} -> classes={classes}")

            fc = _build_head(classes, head, head_conv, pre_feat, final_kernel)
            
            # Head 생성 결과 확인
            if hasattr(fc, '0') and hasattr(fc[0], 'in_channels'):
                print(f"[DLA HEAD BUILD] {head} first layer: in_channels={fc[0].in_channels}, out_channels={fc[0].out_channels}")
            elif hasattr(fc, 'in_channels'):
                print(f"[DLA HEAD BUILD] {head} layer: in_channels={fc.in_channels}, out_channels={fc.out_channels}")
            
            self.__setattr__(head, fc)

    def forward(self, x, save_banks=False, return_stage_backbones=False):
        banks = {} if save_banks else None
        
        # 기존 concat_upper_layer 및 새로운 concat_multi_layers 초기화
        feat_to_cat = None
        feats_to_cat_multi = []
        
        x_base = self.base(x)
        
        # 기존 concat_upper_layer 로직 - base 단계
        if (self.concat_upper_layer is not None) and (self.concat_upper_layer.split('_')[0] == 'base'):
            feat_to_cat = x_base[int(self.concat_upper_layer.split('_')[-1])]
            
        # 새로운 concat_multi_layers 로직 - base 단계에서 수집
        if self.concat_multi_layers is not None:
            for layer_name in self.concat_multi_layers:
                if layer_name.split('_')[0] == 'base':
                    layer_idx = int(layer_name.split('_')[-1])
                    if 0 <= layer_idx < len(x_base):
                        feats_to_cat_multi.append(x_base[layer_idx])
        
        if banks is not None:
            banks.update({'base': x_base})

        # DLA up
        x_dlaup = self.dla_up(x_base)
        
        # DLA up feature map shapes 디버깅 (제거됨)
        
        # 기존 concat_upper_layer 로직 - dlaup 단계
        if (self.concat_upper_layer is not None) and (self.concat_upper_layer.split('_')[0] == 'dlaup'):
            feat_to_cat = x_dlaup[int(self.concat_upper_layer.split('_')[-1])]
            
        # 새로운 concat_multi_layers 로직 - dlaup 단계에서 수집
        if self.concat_multi_layers is not None:
            for layer_name in self.concat_multi_layers:
                if layer_name.split('_')[0] == 'dlaup':
                    layer_idx = int(layer_name.split('_')[-1])
                    if 0 <= layer_idx < len(x_dlaup):
                        feats_to_cat_multi.append(x_dlaup[layer_idx])
        
        if banks is not None:
            banks.update({'dla_up': x_dlaup})

        # IDA up
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x_dlaup[i].clone() if self.clone_to_ida_up else x_dlaup[i])
        
        self.ida_up(y, 0, len(y))
        
        # 기존 concat_upper_layer 로직 - idaup 단계
        if (self.concat_upper_layer is not None) and (self.concat_upper_layer.split('_')[0] == 'idaup'):
            feat_to_cat = y[int(self.concat_upper_layer.split('_')[-1])]
            
        # 새로운 concat_multi_layers 로직 - idaup 단계에서 수집
        if self.concat_multi_layers is not None:
            for layer_name in self.concat_multi_layers:
                if layer_name.split('_')[0] == 'idaup':
                    layer_idx = int(layer_name.split('_')[-1])
                    if 0 <= layer_idx < len(y):
                        feats_to_cat_multi.append(y[layer_idx])
        
        if banks is not None:
            banks.update({'ida_up': y})

        feat_mid = y[-1]
        
        # DLA DEBUG 출력 제거됨

        # ---- Stage 분기 ---- (edit:ccp+ive-fm:25-08-09)
        if self.stage_split or return_stage_backbones:
            feat_low = self.low_from_mid(feat_mid)
        else:
            feat_low = None

        # concat_upper_layer 및 concat_multi_layers 정합 (있다면)
        def _cat(feat_sel, feat_to_cat_var, feats_to_cat_multi_var):
            if feat_sel is None: 
                return None
            
            # 기본 feature로 시작
            result = feat_sel
            
            # 기존 concat_upper_layer 처리
            if self.concat_upper_layer is not None and feat_to_cat_var is not None:
                tgt = feat_to_cat_var
                if result.shape[-2:] != tgt.shape[-2:]:
                    pre = F.interpolate(result, size=tgt.shape[-2:], mode=self.interpolate_mode, 
                                       **_align_kw(self.interpolate_mode, align_corners=False))
                    result = torch.cat([pre, tgt], dim=1)
                else:
                    tgt_resized = F.interpolate(tgt, size=result.shape[-2:], mode=self.interpolate_mode,
                                               **_align_kw(self.interpolate_mode, align_corners=False))
                    result = torch.cat([result, tgt_resized], dim=1)
            
            # 새로운 concat_multi_layers 처리
            if self.concat_multi_layers is not None and len(feats_to_cat_multi_var) > 0:
                for feat_multi in feats_to_cat_multi_var:
                    if result.shape[-2:] != feat_multi.shape[-2:]:
                        # feature를 target size로 resize
                        feat_multi_resized = F.interpolate(feat_multi, size=result.shape[-2:], 
                                                          mode=self.interpolate_mode,
                                                          **_align_kw(self.interpolate_mode, align_corners=False))
                        result = torch.cat([result, feat_multi_resized], dim=1)
                    else:
                        result = torch.cat([result, feat_multi], dim=1)
            
            return result

        feat_mid_backbone = _cat(feat_mid, feat_to_cat, feats_to_cat_multi)
        feat_low_backbone = _cat(feat_low, feat_to_cat, feats_to_cat_multi) if feat_low is not None else None
        
        # Forward shape 확인됨
        

        # head별 stage 선택 (edit:ccp+ive-fm:25-08-09)
        z = {}
        head_stage_map = getattr(self, "_head_stage_map", None)
        for head in self.heads:
            stage = 'mid' if head_stage_map is None else head_stage_map.get(head, 'mid')
            if stage == 'mid' and feat_mid_backbone is not None:
                src = feat_mid_backbone
            else:
                src = feat_low_backbone
            z[head] = getattr(self, head)(src)

        # 반환 (edit:ccp+ive-fm:25-08-09)
        if return_stage_backbones and self.stage_split:
            return z, feat_mid_backbone, feat_low_backbone, banks
        else:
            return z, feat_mid_backbone, banks

