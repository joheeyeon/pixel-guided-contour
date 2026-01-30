import torch.nn as nn
from .snake import Snake
from .utils import (prepare_training, prepare_testing_init, img_poly_to_can_poly, get_gcn_feature,
                    get_gcn_feature_3x3, deterministic_scatter_add, deterministic_scatter_add_vectorized)
import torch
import torch.nn.functional as F


class SeparateTrainableSigmoid(torch.nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        # 클래스별 온도/바이어스(학습 가능)
        self.t_fg = torch.nn.Parameter(torch.tensor(0.0))
        self.t_bg = torch.nn.Parameter(torch.tensor(0.0))
        self.b_fg = torch.nn.Parameter(torch.tensor(0.0))
        self.b_bg = torch.nn.Parameter(torch.tensor(0.0))
        self.eps  = eps

    def forward(self, logits_2ch, detach_logits=True, kappa=0.0):
        # logits_2ch: [B,2,H,W]  (fg, bg)
        if detach_logits:
            logits_2ch = logits_2ch.detach()

        z_bg = logits_2ch[:, 0:1]
        z_fg = logits_2ch[:, 1:2]

        T_fg = F.softplus(self.t_fg) + self.eps
        T_bg = F.softplus(self.t_bg) + self.eps

        s_fg = torch.sigmoid((z_fg - self.b_fg) / T_fg)
        s_bg = torch.sigmoid((z_bg - self.b_bg) / T_bg)

        # print(f"[SeparateTrainableSigmoid] T_fg: {T_fg.item():.4f}, b_fg: {self.b_fg.item():.4f} | T_bg: {T_bg.item():.4f}, b_bg: {self.b_bg.item():.4f}")
        
        # 결과를 concat하여 2채널로 반환
        return torch.cat([s_bg, s_fg], dim=1)

# _ALIGNED_MODES = {"linear", "bilinear", "bicubic", "trilinear"}
#
# def _align_kw(mode: str, align_corners: bool = False):
#     """interpolate용 kwargs: 지원 모드일 때만 align_corners를 넣는다."""
#     return {"align_corners": align_corners} if mode in _ALIGNED_MODES else {}

# class RefinePixel(nn.Module):
#     def __init__(self, dim_out=2, contour_to_map_scale=1., down_ratio=4., num_layers=2, dim_list=[256], kernel_list=[3,1],
#                  dim_in=64):
#         super(RefinePixel, self).__init__()
#         self.sigma = 1.
#         self.contour_to_map_scale = contour_to_map_scale
#         self.down_ratio = down_ratio
#         refine_module_list = []
#         pre_dim = dim_in
#         dim_list = dim_list + [dim_out]
#         for layer_i in range(num_layers):
#             cur_dim = dim_list[layer_i]
#             cur_kernel = kernel_list[layer_i]
#             refine_module_list.append(nn.Conv2d(pre_dim, cur_dim,
#                                                 kernel_size=cur_kernel, padding='same', bias=True))
#             if layer_i < (num_layers - 1):
#                 refine_module_list.append(nn.ReLU(inplace=True))
#             pre_dim = cur_dim
#
#         self.refine_module = nn.Sequential(*refine_module_list)
#
#     def contour_to_map(self, contour, map_size):
#         '''
#         :param contour: NcxNvx2
#         :return: Ncx1xHxW
#         '''
#         # self.sigma = 1.
#         # h, w = map_size
#         # M = torch.zeros([contour.size(0), h, w]).to(contour.device)
#         # x = torch.arange(w).to(contour.device)
#         # y = torch.arange(h).to(contour.device)
#         # grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
#         # grid_x = grid_x.unsqueeze(-1).expand(-1, -1, contour.size(1)).float()
#         # grid_y = grid_y.unsqueeze(-1).expand(-1, -1, contour.size(1)).float()
#         # for i in range(contour.size(0)):
#         #     diff_x = (grid_x - contour[i, ..., 0].unsqueeze(0).unsqueeze(0).float()).pow(2)
#         #     diff_y = (grid_y - contour[i, ..., 1].unsqueeze(0).unsqueeze(0).float()).pow(2)
#         #     M[i, ...] += torch.sum(torch.exp(-(diff_x + diff_y) / (2 * self.sigma ** 2)), -1)
#         # M /= (2 * torch.pi * self.sigma ** 2)
#         # from scipy.io import savemat
#         # savemat(f"data/test.mat",
#         #         {'M': M.detach().cpu().numpy(), 'grid_x': grid_x.detach().cpu().numpy(), 'grid_y': grid_y.detach().cpu().numpy(), 'contour': contour.detach().cpu().numpy()})
#         h, w = map_size
#         device = contour.device
#
#         # GPU 메모리 최적화를 위해 필요할 때만 연산
#         x = torch.arange(w, device=device).view(1, w, 1).float()
#         y = torch.arange(h, device=device).view(h, 1, 1).float()
#
#         M = torch.zeros((contour.size(0), h, w), device=device)
#
#         for i in range(contour.size(0)):  # 배치별 연산
#             cx = contour[i, :, 0].view(1, 1, -1)  # (1, 1, Nv)
#             cy = contour[i, :, 1].view(1, 1, -1)  # (1, 1, Nv)
#
#             diff_x = (x - cx).pow(2)  # (1, W, Nv)
#             diff_y = (y - cy).pow(2)  # (H, 1, Nv)
#
#             M[i] = torch.exp(-(diff_x + diff_y) / (2 * self.sigma ** 2)).sum(dim=-1)
#
#         M /= (2 * torch.pi * self.sigma ** 2)
#         return M * self.contour_to_map_scale
#
#     def forward(self, contour, feature, batch_ind, pre_pixel_map=None):
#         contour_map = self.contour_to_map(contour/self.down_ratio, feature.shape[2:])
#         # Nx1xHxW 형태로 변환할 텐서 초기화
#         contour_map_b = torch.zeros(feature.size(0), feature.size(2), feature.size(3)).to(feature.device)
#         # masks를 일차원으로 펼친 후 scatter_add를 이용해 합산
#         contour_map_b = contour_map_b.scatter_add(0, batch_ind.view(-1, 1, 1).expand(-1, feature.size(2), feature.size(3)).to(dtype=torch.int64), contour_map)
#         if pre_pixel_map is not None:
#             feature_cat = torch.cat([pre_pixel_map, contour_map_b.unsqueeze(1), feature], dim=1) # !to each img_idx
#         else:
#             feature_cat = torch.cat([contour_map_b.unsqueeze(1), feature], dim=1)  # !to each img_idx
#         pixelmap_refined = self.refine_module(feature_cat)
#         return pixelmap_refined, contour_map_b
class RefinePixel(nn.Module):
    def __init__(self, dim_out=2, contour_to_map_scale=1., down_ratio=4., num_layers=2, dim_list=[256],
                 kernel_list=[3, 1], dim_in=64, input_norm_type='unnormalized', reduce_memory=False,
                 refine_as_residual=False, convert_map_down_ratio=2, module_structure=None, contour_map_shape=[104,104],
                 deterministic="full"):
        super(RefinePixel, self).__init__()
        self.sigma = 1.
        self.contour_to_map_scale = contour_to_map_scale
        self.down_ratio = down_ratio
        self.input_norm_type = input_norm_type
        self.reduce_memory = reduce_memory
        self.refine_as_residual = refine_as_residual
        self.convert_map_down_ratio = convert_map_down_ratio
        self.contour_map_shape = contour_map_shape
        self.deterministic = deterministic
        # self.contour_map_b = torch.zeros([1, 1] + contour_map_shape)

        if module_structure is not None:
            final_kernel = 1
            self.refine_module = nn.Sequential()
            pre_feat = dim_in
            for layer in module_structure.keys():
                if layer.split('_')[0] == 'conv':
                    for convi in range(len(module_structure[layer]) if isinstance(module_structure[layer], list) else 1):
                        n_feat = module_structure[layer][convi] if isinstance(module_structure[layer], list) else module_structure[layer]
                        self.refine_module.add_module(f"conv_{layer.split('_')[-1]}_{convi}",
                                      nn.Conv2d(pre_feat, n_feat, kernel_size=3, padding=1, bias=True))
                        self.refine_module.add_module(f"relu_{layer.split('_')[-1]}_{convi}", nn.ReLU(inplace=True))
                        pre_feat = n_feat
                elif layer.split('_')[0] == 'up':
                    self.refine_module.add_module(f"up_{layer.split('_')[-1]}",
                                  nn.Upsample(scale_factor=module_structure[layer], mode=layer.split('_')[1]))
                else:  # layer.split('_')[0] == 'convtranspose':
                    n_feat = module_structure[layer][-1]
                    self.refine_module.add_module(f"convtranspose_{layer.split('_')[-1]}",
                                  nn.ConvTranspose2d(pre_feat, n_feat, kernel_size=module_structure[layer][0],
                                                     stride=module_structure[layer][0]))
                    self.refine_module.add_module(f"relutranspose_{layer.split('_')[-1]}", nn.ReLU(inplace=True))
                    pre_feat = n_feat

            self.refine_module.add_module(f"conv_out", nn.Conv2d(pre_feat, dim_out, kernel_size=final_kernel, stride=1,
                                                 padding=final_kernel // 2, bias=True))
        else:
            refine_module_list = []
            pre_dim = dim_in
            dim_list = dim_list + [dim_out]
            for layer_i in range(num_layers):
                cur_dim = dim_list[layer_i]
                cur_kernel = kernel_list[layer_i]
                refine_module_list.append(nn.Conv2d(pre_dim, cur_dim, kernel_size=cur_kernel, padding='same', bias=True))
                if layer_i < (num_layers - 1):
                    refine_module_list.append(nn.ReLU(inplace=True))
                pre_dim = cur_dim

            self.refine_module = nn.Sequential(*refine_module_list)

    def contour_to_map(self, contour, map_size):
        h, w = map_size
        device = contour.device

        if contour.shape[0] == 0:  # batch가 0일 때 체크
            return None

        Nc, Nv, _ = contour.shape
        
        if self.reduce_memory:
            # 해상도 줄이기 (원래 설정 유지)
            h_map, w_map = h // self.convert_map_down_ratio, w // self.convert_map_down_ratio
        else:
            h_map, w_map = h, w

        y = torch.arange(h_map, device=device).view(h_map, 1, 1).float()
        x = torch.arange(w_map, device=device).view(1, w_map, 1).float()

        if self.reduce_memory:
            cx = (contour[:, :, 0] / self.convert_map_down_ratio).unsqueeze(1).unsqueeze(1)
            cy = (contour[:, :, 1] / self.convert_map_down_ratio).unsqueeze(1).unsqueeze(1)
        else:
            cx = contour[:, :, 0].unsqueeze(1).unsqueeze(1)
            cy = contour[:, :, 1].unsqueeze(1).unsqueeze(1)

        # ✅ 청킹 제거 - 직접 연산으로 성능 향상
        diff_x = (x - cx).pow(2)
        diff_y = (y - cy).pow(2)
        dist = diff_x + diff_y
        
        M = torch.exp(-dist / (2 * self.sigma ** 2)).sum(dim=-1)
        M /= (2 * torch.pi * self.sigma ** 2)

        if self.reduce_memory:
            # 업샘플링 (원래 방식 유지)
            if self.deterministic == "full":
                with torch.no_grad():
                    M = F.interpolate(M.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
            else:
                M = F.interpolate(M.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

        result = M * self.contour_to_map_scale
        return result
    
    def _process_contour_batch(self, contour, h_map, w_map, h, w, device):
        """배치 단위로 contour를 처리하는 헬퍼 함수"""
        Nc = contour.shape[0]
        
        # ✅ 더 작은 배치로 재귀 분할 (극한 메모리 절약)
        if Nc > 5:  # 5개씩 더 작은 단위로 처리
            results = []
            for i in range(0, Nc, 5):
                end_i = min(i + 5, Nc)
                mini_batch = contour[i:end_i]
                mini_result = self._process_contour_batch(mini_batch, h_map, w_map, h, w, device)
                if mini_result is not None:
                    results.append(mini_result)
                # 메모리 정리는 학습 성능에 영향을 줄 수 있으므로 제거
            
            if results:
                return torch.cat(results, dim=0)
            else:
                return None
        
        # 실제 처리 (5개 이하만)
        y = torch.arange(h_map, device=device).view(h_map, 1, 1).float()
        x = torch.arange(w_map, device=device).view(1, w_map, 1).float()

        if self.reduce_memory:
            cx = (contour[:, :, 0] / self.convert_map_down_ratio).unsqueeze(1).unsqueeze(1)
            cy = (contour[:, :, 1] / self.convert_map_down_ratio).unsqueeze(1).unsqueeze(1)
        else:
            cx = contour[:, :, 0].unsqueeze(1).unsqueeze(1)
            cy = contour[:, :, 1].unsqueeze(1).unsqueeze(1)

        # ✅ 청킹 완전 제거 - 직접 연산으로 성능 향상
        diff_x = (x - cx).pow(2)
        diff_y = (y - cy).pow(2)
        dist = diff_x + diff_y
        M = torch.exp(-dist / (2 * self.sigma ** 2)).sum(dim=-1)
        M /= (2 * torch.pi * self.sigma ** 2)

        if self.reduce_memory:
            # 업샘플링
            if self.deterministic == "full":
                with torch.no_grad():
                    M = F.interpolate(M.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
            else:
                M = F.interpolate(M.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

        result = M * self.contour_to_map_scale
        del M
        return result

    # def contour_to_map(self, contour, map_size):
    #     '''
    #     contour: (Nc, Nv, 2)
    #     return: (Nc, H, W)
    #     '''
    #     h, w = map_size
    #     device = contour.device
    #     Nc, Nv, _ = contour.shape
    #
    #     # (H, 1, 1)
    #     y = torch.arange(h, device=device).view(h, 1, 1).float()
    #     # (1, W, 1)
    #     x = torch.arange(w, device=device).view(1, w, 1).float()
    #
    #     # (Nc, Nv) → (Nc, 1, 1, Nv)
    #     cx = contour[:, :, 0].unsqueeze(1).unsqueeze(1)
    #     cy = contour[:, :, 1].unsqueeze(1).unsqueeze(1)
    #
    #     # broadcasting (Nc, H, W, Nv)
    #     diff_x = (x - cx).pow(2)  # (Nc, H, W, Nv)
    #     diff_y = (y - cy).pow(2)
    #
    #     dist = diff_x + diff_y  # (Nc, H, W, Nv)
    #
    #     M = torch.exp(-dist / (2 * self.sigma ** 2)).sum(dim=-1)  # sum over Nv
    #
    #     M /= (2 * torch.pi * self.sigma ** 2)
    #
    #     return M * self.contour_to_map_scale  # (Nc, H, W)

    def forward(self, contour, feature, batch_ind, pre_pixel_map=None):
        '''
        contour: (Nc, Nv, 2)
        feature: (N, C, H, W)
        batch_ind: (Nc,)
        pre_pixel_map: (optional) (N, 1, H, W)
        '''
        device = feature.device
        N, _, H, W = feature.shape

        # contour (pixel 위치) → 2D gaussian map
        contour_map = self.contour_to_map(contour / self.down_ratio, (H, W))  # (Nc, H, W)

        # 작업용 버퍼
        if (not hasattr(self, 'contour_map_b_buffer')) or \
                (self.contour_map_b_buffer.shape != (N, 1, H, W)) or \
                (self.contour_map_b_buffer.device != device):
            self.contour_map_b_buffer = torch.zeros((N, 1, H, W), device=device)
        if self.contour_map_b_buffer.requires_grad:
            self.contour_map_b_buffer = self.contour_map_b_buffer.detach()
            self.contour_map_b_buffer.requires_grad_(False)

        self.contour_map_b_buffer.zero_()

        if contour_map is None:
            scattered_result = self.contour_map_b_buffer.clone()
        else:
            # idx = batch_ind.view(-1, 1, 1).expand(-1, H, W).to(dtype=torch.long)  # (Nc, H, W)
            # # contour_map_b = contour_map_b.scatter_add(0, idx, contour_map)
            # # print(f"contour_map shape : {contour_map.shape}")
            # # contour_map_b = deterministic_scatter_add(contour_map, idx, dim=0, out=contour_map_b.squeeze(1))
            # # 🎯 out은 (N, H, W)여야 하므로 buffer의 view 사용
            # out_2d = self.contour_map_b_buffer.view(N, H, W)
            # deterministic_scatter_add(contour_map.clone(), idx, out_2d)
            #
            # contour_map_b = self.contour_map_b_buffer  # 그대로 반환 (N, 1, H, W)
            # 1. 텐서 모양을 (Batch, Features) 형태로 변경합니다.
            # src: (Nc, H, W) -> (Nc, H*W)
            src_reshaped = contour_map.view(contour_map.shape[0], -1)

            # --- scatter 결과 만들기 ---
            if contour_map is None:
                scattered_result = self.contour_map_b_buffer.clone()
            else:
                if self.deterministic == "full":
                    src_reshaped = contour_map.view(contour_map.shape[0], -1)
                    scattered_result = deterministic_scatter_add_vectorized(
                        src=src_reshaped, index=batch_ind, dim_size=N
                    ).view(N, 1, H, W)  # 새 텐서
                else:
                    out_2d = self.contour_map_b_buffer.view(N, H, W)
                    out_2d.scatter_add_(
                        dim=0,
                        index=batch_ind.view(-1, 1, 1).expand(-1, H, W).long(),
                        src=contour_map
                    )
                    scattered_result = out_2d.unsqueeze(1).clone()  # alias 차단
            scattered_result = scattered_result.contiguous()

        if pre_pixel_map is not None:
            if self.input_norm_type == 'softmax':
                pre_pixel_map = F.softmax(pre_pixel_map, dim=1)
            elif self.input_norm_type == 'argmax':
                pre_pixel_map = pre_pixel_map.argmax(dim=1, keepdim=True).float()

            if pre_pixel_map.shape[-2:] != (H, W):
                pre_pixel_map = F.interpolate(pre_pixel_map, size=(H, W), mode='bilinear', align_corners=False)

            feature_cat = torch.cat([pre_pixel_map, scattered_result, feature], dim=1)  # (N, 1+1+C, H, W)
        else:
            feature_cat = torch.cat([scattered_result, feature], dim=1)  # (N, 1+C, H, W)

        module_out = self.refine_module(feature_cat)

        if self.refine_as_residual and pre_pixel_map is not None:
            pixelmap_refined = pre_pixel_map + module_out
        else:
            pixelmap_refined = module_out

        return pixelmap_refined, scattered_result.squeeze(1)


class Evolution(nn.Module):
    def __init__(self, evolve_iter_num=1, evolve_stride=1., ro=4., with_img_idx=False, refine_kernel_size=3, in_featrue_dim=64, channel_pixel=2, use_vertex_classifier=False,
                 c_out_proj=64, with_proj=True, num_vertex=128, pixel_norm_type='unnormalized', vtx_cls_common_prediction_type=1, vtx_cls_kernel_size=1, cfg=None, gcn_weight_sharing=True,
                 use_3x3_feature=False, feature_3x3_mode='flatten'):
        super(Evolution, self).__init__()
        assert evolve_iter_num >= 1
        self.use_vertex_classifier = use_vertex_classifier
        self.evolve_stride = evolve_stride
        self.ro = ro
        self.iter = evolve_iter_num
        self.with_img_idx = with_img_idx
        self.with_proj = with_proj
        self.num_vertex = num_vertex
        self.pixel_norm_type = pixel_norm_type
        self.vtx_cls_common_prediction_type = vtx_cls_common_prediction_type
        self.vtx_cls_kernel_size = vtx_cls_kernel_size
        self.cfg = cfg  # ✅ cfg 저장
        self.gcn_weight_sharing = gcn_weight_sharing  # ✅ weight sharing 옵션 저장
        self.use_3x3_feature = use_3x3_feature  # ✅ 3x3 feature 사용 여부
        self.feature_3x3_mode = feature_3x3_mode  # ✅ 3x3 feature 모드
        
        # trainable temperature parameter for softmax
        if self.pixel_norm_type == 'trainable_softmax':
            # 각 채널별 temperature 파라미터 - softplus의 역함수로 초기화
            # softplus_inverse(1.0) ≈ 0.54로 초기화하여 초기 temperature = 1.0
            init_val = torch.log(torch.exp(torch.tensor(1.0)) - 1)  # softplus_inverse(1.0)
            self.temperature = nn.Parameter(torch.full((channel_pixel,), init_val.item()))  # learnable temperature parameter per channel
        elif self.pixel_norm_type == 'trainable_softmax_softclamp':
            # sigmoid를 사용한 soft clamping으로 temperature 범위 제한
            import math
            C = channel_pixel
            self.T_lo, self.T_hi = 0.5, 4.0   # 넓고 안전한 가드레일
            T_init = 1.0
            
            # u는 자유파라미터(실제 학습되는 값)
            self.u = nn.Parameter(torch.zeros(C))
            
            # T_init에 맞춰 u 초기화 (sigmoid(u0) = p)
            p = (T_init - self.T_lo) / (self.T_hi - self.T_lo)  # 0~1
            u0 = math.log(p / (1.0 - p))  # sigmoid_inverse(p)
            with torch.no_grad():
                self.u.fill_(u0)
            print(f"[trainable_softmax_softclamp] Init: T_lo={self.T_lo}, T_hi={self.T_hi}, T_init={T_init}, u0={u0:.4f}")
        # separate trainable sigmoid for fg/bg
        elif self.pixel_norm_type == 'sep_trainable_sigmoid':
            self.sep_sigmoid = SeparateTrainableSigmoid()

        if self.with_proj:
            self.c_out_proj = c_out_proj
            self.projection = torch.nn.Sequential(torch.nn.Conv2d(in_featrue_dim + channel_pixel, 256, kernel_size=3,
                                                                     padding=1, bias=True),
                                                     torch.nn.ReLU(inplace=True),
                                                     torch.nn.Conv2d(256, self.c_out_proj, kernel_size=1,
                                                                     stride=1, padding=0, bias=True))
        else:
            self.c_out_proj = in_featrue_dim + channel_pixel
            self.projection = None

        # ✅ weight sharing 여부에 따라 GCN 모듈 생성 방식 결정
        # 3x3 flatten mode일 때는 feature가 이미 C*9로 flatten되어 들어옴
        if self.use_3x3_feature and self.feature_3x3_mode == 'flatten':
            snake_feature_dim = self.c_out_proj * 9 + 2  # (66*9) + 2 = 596
        else:
            snake_feature_dim = self.c_out_proj + 2  # 66 + 2 = 68
        
        if self.gcn_weight_sharing:
            # 기존 방식: 하나의 GCN 모듈을 모든 iteration에서 공유
            self.evolve_gcn = Snake(state_dim=num_vertex, feature_dim=snake_feature_dim, conv_type='dgrid',
                                    with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size,
                                    use_vertex_classifier=self.use_vertex_classifier, common_prediction_type=self.vtx_cls_common_prediction_type,
                                    vtx_cls_kernel_size=self.vtx_cls_kernel_size,
                                    use_3x3_feature=self.use_3x3_feature, feature_3x3_mode=self.feature_3x3_mode)
            self.evolve_gcn_list = None
        else:
            # 새로운 방식: iteration마다 독립적인 GCN 모듈 사용 (non-sharing)
            self.evolve_gcn = None
            self.evolve_gcn_list = nn.ModuleList([
                Snake(state_dim=num_vertex, feature_dim=snake_feature_dim, conv_type='dgrid',
                      with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size,
                      use_vertex_classifier=self.use_vertex_classifier, common_prediction_type=self.vtx_cls_common_prediction_type,
                      vtx_cls_kernel_size=self.vtx_cls_kernel_size,
                      use_3x3_feature=self.use_3x3_feature, feature_3x3_mode=self.feature_3x3_mode)
                for _ in range(self.iter)
            ])

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch, num_points=128):
        init = prepare_training(output, batch, self.ro, with_img_idx=self.with_img_idx, num_points=num_points)
        return init

    def prepare_testing_init(self, output, num_points_init=128):
        if self.with_img_idx and output['ct_01'].size(1) > 0:
            polys = output['poly_coarse'][output['ct_01']]
        else:
            polys = output['poly_coarse']
        init = prepare_testing_init(polys, self.ro, num_points=num_points_init, py_ind=output['py_ind'] if 'py_ind' in output else output['batch_ind'])
        return init

    def prepare_testing_evolve(self, output, h, w, cfg=None):
        img_init_polys = output['img_init_polys']
        # ✅ ccp_maskinit에서는 clip_image 비활성화
        if cfg is None or cfg.commen.task != 'ccp_maskinit':
            img_init_polys[..., 0] = torch.clamp(img_init_polys[..., 0], min=0, max=w-1)
            img_init_polys[..., 1] = torch.clamp(img_init_polys[..., 1], min=0, max=h-1)
        output.update({'img_init_polys': img_init_polys})
        return img_init_polys
    
    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, stride=1., ignore=False, extract_offset=False, ct_01=None, return_vertex_classifier=False, cfg=None, iter_idx=0):
        if ignore:
            # ✨ 스킵하더라도 collective 횟수 맞추기 (edit:ddp-sync-bn-dummy:25-08-09)
            snake.run_dummy_like(cnn_feature)
            if extract_offset:
                if return_vertex_classifier:
                    return i_it_poly * self.ro, torch.empty(i_it_poly.size(),device=i_it_poly.device), i_it_poly * 0.0
                else:
                    return i_it_poly * self.ro, i_it_poly * 0.0
            else:
                if return_vertex_classifier:
                    return i_it_poly * self.ro, torch.empty(i_it_poly.size(),device=i_it_poly.device)
                else:
                    return i_it_poly * self.ro
        if len(i_it_poly) == 0:
            # import torch.distributed as dist
            # if dist.is_initialized():
            #     r = dist.get_rank()
            #     print(f"[DEBUG] [R{r}] DUMMY snake at evolve_poly (ignore={ignore}, len={len(i_it_poly)})", flush=True)
            # ✨ 여기서도 동일하게 collective 맞춤 (edit:ddp-sync-bn-dummy:25-08-09)
            snake.run_dummy_like(cnn_feature)
            # 🚨 DDP-safe forward pass:
            #   입력이 비어있을 때, 모델 파라미터와 연결된 0 값의 텐서를 반환하여
            #   DDP 환경에서 모든 GPU의 연산 그래프를 동일하게 유지합니다.
            # dummy_val = 0.0 * sum(p.sum() for p in snake.parameters())
            dummy_poly = torch.zeros_like(i_it_poly)
            if extract_offset:
                if return_vertex_classifier:
                    dummy_logits = torch.zeros(i_it_poly.size(0), 2, i_it_poly.size(1),
                                               device=i_it_poly.device, dtype=i_it_poly.dtype)
                    return dummy_poly, dummy_logits, dummy_poly
                else:
                    return dummy_poly, dummy_poly
            else:
                if return_vertex_classifier:
                    dummy_logits = torch.zeros(i_it_poly.size(0), 2, i_it_poly.size(1),
                                               device=i_it_poly.device, dtype=i_it_poly.dtype)
                    return dummy_poly, dummy_logits
                else:
                    return dummy_poly
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        if self.with_img_idx:
            i_it_poly_compact = i_it_poly[ct_01]
        else:
            i_it_poly_compact = i_it_poly

        # print(f"i_it_poly_compact : {i_it_poly_compact.shape}, ind : {ind.shape}")
        # print(f"[DEBUG] i_it_poly_compact : {i_it_poly_compact.max()}, in {h}x{w}")
        
        # Use 3x3 feature extraction if enabled
        if self.use_3x3_feature:
            # Get detach option from config
            detach_features = getattr(self.cfg.model, 'feature_3x3_detach', False) if self.cfg else False
            init_feature = get_gcn_feature_3x3(cnn_feature, i_it_poly_compact, ind, h, w, 
                                             mode=self.feature_3x3_mode, detach_features=detach_features)
        else:
            init_feature = get_gcn_feature(cnn_feature, i_it_poly_compact, ind, h, w)
            
        c_it_poly = c_it_poly * self.ro
        if self.with_img_idx:
            c_it_poly_reshape = c_it_poly.permute(0, 3, 2, 1) #(n_im, 2, n_vert, n_ct)
            
            # Handle different feature dimensions based on 3x3 mode
            if self.use_3x3_feature and self.feature_3x3_mode == 'conv2d':
                # For conv2d mode with img_idx - simplify by not supporting for now
                # Convert to flatten mode behavior temporarily
                N_poly, C, N_vert, H, W = init_feature.shape
                init_feature_flat = init_feature.view(N_poly, C*H*W, N_vert)  # flatten spatial
                init_feature_reshape = torch.zeros([c_it_poly.size(0),c_it_poly.size(1),C*H*W,N_vert]).to(c_it_poly.device) #(n_im, n_ct, C*9, n_vert)
                init_feature_reshape[ct_01] = init_feature_flat #(n_py, C*9, n_vert)
                init_feature_reshape = init_feature_reshape.permute(0, 2, 3, 1) #(n_im, C*9, n_vert, n_ct)
            elif self.use_3x3_feature and self.feature_3x3_mode == 'spatial':
                # init_feature: [N_poly, C, N_vert, 3, 3] -> need special handling
                # For now, create zeros with correct shape based on spatial feature
                init_feature_reshape = torch.zeros([c_it_poly.size(0), c_it_poly.size(1), 
                                                   init_feature.size(1), init_feature.size(2), 
                                                   init_feature.size(3), init_feature.size(4)]).to(c_it_poly.device)
                init_feature_reshape[ct_01] = init_feature  # [n_py, C, n_vert, 3, 3]
                # For concat, need to reshape to match c_it_poly_reshape dimensions
                # This is complex - for now, let's flatten the spatial dimensions
                N_im, N_ct, C, N_vert, H, W = init_feature_reshape.shape
                init_feature_reshape = init_feature_reshape.view(N_im, N_ct, C*H*W, N_vert)
                init_feature_reshape = init_feature_reshape.permute(0, 2, 3, 1) #(n_im, C*H*W, n_vert, n_ct)
            else:
                # Standard case: [N_poly, C, N_vert] or [N_poly, C*9, N_vert]
                init_feature_reshape = torch.zeros([c_it_poly.size(0),c_it_poly.size(1),init_feature.size(1),init_feature.size(2)]).to(c_it_poly.device) #(n_im, n_ct, n_feat, n_vert)
                init_feature_reshape[ct_01] = init_feature #(n_py, n_feat, n_vert)
                init_feature_reshape = init_feature_reshape.permute(0, 2, 3, 1) #(n_im, n_feat, n_vert, n_ct)
            
            # For with_img_idx case, always concatenate in the standard way
            init_input = torch.cat([init_feature_reshape, c_it_poly_reshape], dim=1)
        else:
            # Handle different feature dimensions for non-img_idx case
            if self.use_3x3_feature and self.feature_3x3_mode == 'spatial':
                # init_feature: [N_poly, C, N_vert, 3, 3] -> flatten to [N_poly, C*9, N_vert]
                N_poly, C, N_vert, H, W = init_feature.shape
                init_feature_reshape = init_feature.view(N_poly, C*H*W, N_vert)
                c_it_poly_reshape = c_it_poly.permute(0, 2, 1)
                init_input = torch.cat([init_feature_reshape, c_it_poly_reshape], dim=1)
            elif self.use_3x3_feature and self.feature_3x3_mode == 'conv2d':
                # For conv2d mode, pass 5D feature directly to Snake, let it handle the processing
                init_feature_reshape = init_feature  # [N_poly, C, N_vert, 3, 3]
                c_it_poly_reshape = c_it_poly.permute(0, 2, 1)  # [N_poly, 2, N_vert]
                # Don't concatenate here - Snake will handle it internally
                init_input = (init_feature_reshape, c_it_poly_reshape)  # Pass as tuple
            else:
                # Standard case: [N_poly, C, N_vert] or [N_poly, C*9, N_vert]
                init_feature_reshape = init_feature
                c_it_poly_reshape = c_it_poly.permute(0, 2, 1)
                init_input = torch.cat([init_feature_reshape, c_it_poly_reshape], dim=1)
        # import torch.distributed as dist
        # if dist.is_initialized():
        #     r = dist.get_rank()
        #     print(f"[DEBUG] [R{r}] CALL snake at evolve_poly (ignore={ignore}, len={len(i_it_poly)})", flush=True)
        if return_vertex_classifier:
            offset, valid_logits = snake(init_input, return_vertex_classifier)
        else:
            offset = snake(init_input)
            valid_logits = None
        if self.with_img_idx:
            offset = offset.permute(0, 3, 2, 1) #(n_im, 2, n_vert, n_ct) > (n_im, n_ct, n_vert, 2)
        else:
            offset = offset.permute(0, 2, 1)

        # ✅ ccp_maskinit에서는 논리적으로 맞는 방식 사용: (i_it_poly + offset) * self.ro
        if cfg is not None and hasattr(cfg, 'commen') and hasattr(cfg.commen, 'task') and cfg.commen.task == 'ccp_maskinit':
            i_poly = (i_it_poly.detach() + offset * stride) * self.ro
        elif hasattr(self, 'cfg') and self.cfg is not None and hasattr(self.cfg, 'commen') and hasattr(self.cfg.commen, 'task') and self.cfg.commen.task == 'ccp_maskinit':
            i_poly = (i_it_poly.detach() + offset * stride) * self.ro
        else:
            # 기존 학습된 모델들은 원래 방식 그대로 유지 (학습된 가중치에 맞춰)
            i_poly = i_it_poly.detach() * self.ro + offset * stride
        if extract_offset:
            if return_vertex_classifier:
                return i_poly, valid_logits, offset * stride
            else:
                return i_poly, offset * stride
        else:
            if return_vertex_classifier:
                return i_poly, valid_logits
            else:
                return i_poly

    def foward_train(self, output, batch, cnn_feature, return_vertex_classifier=False, num_points_init=128):
        ret = output
        # if self.with_img_idx:
        #     # print(f"init['img_init_polys'] shape: {init['img_init_polys'].shape}")
        #     # print(f"init['ct_01'] shape : {init['ct_01'].shape}")
        #     # i_it_poly = torch.zeros([init['ct_01'].size(0), init['ct_01'].size(1), init['img_init_polys'].size(1), init['img_init_polys'].size(2)]).to(init['img_init_polys'].device)
        #     # i_it_poly[init['ct_01']] = init['img_init_polys']
        #     i_it_poly = init['img_init_polys']
        #     c_it_poly = torch.zeros_like(i_it_poly)
        #     c_it_poly[init['ct_01']] = init['can_init_polys']
        # else:
        if 'py_pred' in ret:
            py_preds = ret['py_pred']
            py_pred = py_preds[-1] / self.ro
            c_py_pred = img_poly_to_can_poly(py_pred)
            py_ind = ret['batch_ind']
            ct_01 = ret['ct_01']
        else:
            init = self.prepare_training(output, batch, num_points=num_points_init)
            py_pred = init['img_init_polys']
            c_py_pred = init['can_init_polys']
            py_preds = []
            py_ind = init['py_ind']
            ct_01 = init['ct_01']

        # print(f"[DEBUG] py_pred : {py_pred.shape}")
        # if torch.numel(py_pred)> 0:
        #     print(f"[DEBUG] py_pred : {py_pred.min()}, {py_pred.max()}")
        # print(f"pre py_pred : {py_pred.requires_grad}")

        for i in range(self.iter):
            # print(f"{i}")
            if self.pixel_norm_type == 'softmax':
                pixel_map = F.softmax(output['pixel'][-1], dim=1)
            elif self.pixel_norm_type == 'trainable_softmax':
                # 각 채널별 temperature 적용: exp(logit/temperature_c)
                logits = output['pixel'][-1]  # [B, C, H, W]
                # temperature를 양수로 보장 (softplus 또는 clamp 사용)
                positive_temp = F.softplus(self.temperature) + 1e-8  # 최소값 보장
                temp_reshaped = positive_temp.view(1, -1, 1, 1)
                pixel_map = F.softmax(logits / temp_reshaped, dim=1)
            elif self.pixel_norm_type == 'trainable_softmax_softclamp':
                # sigmoid를 사용한 soft clamping으로 temperature 범위 제한
                logits = output['pixel'][-1]  # [B, C, H, W]
                # T = T_lo + (T_hi - T_lo) * sigmoid(u)  ∈ [T_lo, T_hi]
                T = self.T_lo + (self.T_hi - self.T_lo) * torch.sigmoid(self.u)  # [C]
                temp_reshaped = T.view(1, -1, 1, 1)
                pixel_map = F.softmax(logits / temp_reshaped, dim=1)
            elif self.pixel_norm_type == 'sep_trainable_sigmoid':
                pixel_map = self.sep_sigmoid(output['pixel'][-1])
            elif self.pixel_norm_type == 'argmax':
                pixel_map = torch.argmax(output['pixel'][-1], dim=1).unsqueeze(1)
            else:
                pixel_map = output['pixel'][-1]

            # if pixel_map.size(2)!=cnn_feature.size(2) or pixel_map.size(3)!=cnn_feature.size(3):
                # with torch.backends.cudnn.flags(enabled=True, deterministic=False):
                # with torch.no_grad(): #edit to no_grad (for reprod)
                # pixel_map = F.interpolate(pixel_map, size=(cnn_feature.size(2), cnn_feature.size(3)), mode='nearest')
                # print(f"pixel_map : {pixel_map.shape}, cnn_feature : {cnn_feature.shape}")
                # if self.training:
                #     with torch.backends.cudnn.flags(enabled=True, deterministic=False):
                #         pixel_map = F.interpolate(pixel_map, size=(cnn_feature.size(2), cnn_feature.size(3)), mode='bilinear', align_corners=False).squeeze(1)
                # else:
                #     with torch.no_grad():
                #         pixel_map = F.interpolate(pixel_map, size=(cnn_feature.size(2), cnn_feature.size(3)), mode='bilinear', align_corners=False).squeeze(1)

            if self.projection is not None:
                feature_combined = self.projection(torch.cat([pixel_map, cnn_feature], dim=1))
            else:
                feature_combined = torch.cat([pixel_map, cnn_feature], dim=1)
            # ✅ weight sharing 여부에 따라 적절한 GCN 선택
            current_gcn = self.evolve_gcn if self.gcn_weight_sharing else self.evolve_gcn_list[i]
            
            if (self.use_vertex_classifier) and return_vertex_classifier:
                py_pred, py_isvalid = self.evolve_poly(current_gcn, feature_combined, py_pred, c_py_pred,
                                           py_ind, stride=self.evolve_stride, ct_01=ct_01,
                                           return_vertex_classifier=True, cfg=self.cfg, iter_idx=i)
                if py_isvalid is not None:
                    if 'py_valid_logits' in ret:
                        pre_py_valid_logits = ret.get('py_valid_logits', [])
                        pre_py_valid_logits.append(py_isvalid)
                        ret.update({'py_valid_logits': pre_py_valid_logits})
                    else:
                        ret.update({'py_valid_logits': [py_isvalid]})

            else:
                py_pred = self.evolve_poly(current_gcn, feature_combined, py_pred, c_py_pred,
                                           py_ind, stride=self.evolve_stride, ct_01=ct_01, cfg=self.cfg, iter_idx=i)

            # print(f"py_pred : {py_pred.min()}, {py_pred.max()}")
            py_preds.append(py_pred)
            ret.setdefault('py_keys', []).append('deform')
            
            py_pred = py_pred / self.ro
            c_py_pred = img_poly_to_can_poly(py_pred)
        ret.update({'py_pred': py_preds})
        if 'img_gt_polys' not in ret:
            ret.update({'img_gt_polys': init['img_gt_polys'] * self.ro})
        if  'img_gt_init_polys' not in ret:
            ret.update({'img_gt_init_polys': init['img_gt_init_polys']*self.ro})
        if 'img_gt_coarse_polys' not in ret:
            ret.update({'img_gt_coarse_polys': init['img_gt_coarse_polys']*self.ro})
        if 'batch_ind' not in ret:
            ret.update({'batch_ind': py_ind.to(device=py_pred.device)})
        if 'ct_01' not in ret:
            ret.update({'ct_01': ct_01})
        return output

    def foward_test(self, output, cnn_feature, ignore, extract_offset=False, num_points_init=128, return_vertex_classifier=False):
        ret = output
        with torch.no_grad():
            # if extract_offset:
            #     py, py_offset = self.evolve_poly(self.evolve_gcn, cnn_feature, i_it_poly, c_it_poly,
            #                                      init['py_ind'],
            #                                      ignore=ignore[0], stride=self.evolve_stride, extract_offset=True, ct_01=ct_01)
            # else:
            #     py = self.evolve_poly(self.evolve_gcn, cnn_feature, i_it_poly, c_it_poly, init['py_ind'],
            #                           ignore=ignore[0], stride=self.evolve_stride, ct_01=ct_01)
            # if ct_01 is None:
            #     pys = [py, ]
            # elif ct_01.numel() > 0:
            #     pys = [py[ct_01], ]
            # else:
            #     pys = [py, ]
            # if extract_offset:
            #     pys_offset = [py_offset, ]
            # else:
            #     pys_offset = None
            if 'py' in ret:
                pys = ret['py']
                py = pys[-1] / self.ro
                c_py = img_poly_to_can_poly(py)
                py_ind = ret['batch_ind']
                ct_01 = ret['ct_01']
            else:
                init = self.prepare_testing_init(output, num_points_init=num_points_init)
                img_init_polys = self.prepare_testing_evolve(init, cnn_feature.size(2), cnn_feature.size(3), cfg=self.cfg)
                if self.with_img_idx:
                    ct_01 = torch.ones([cnn_feature.size(0), img_init_polys.size(0)], dtype=torch.bool)
                    py = torch.zeros([ct_01.size(0), ct_01.size(1), img_init_polys.size(1),
                                      img_init_polys.size(2)]).to(img_init_polys.device)
                    py[ct_01] = img_init_polys
                    c_py = torch.zeros_like(py)
                    c_py[ct_01] = init['can_init_polys']
                else:
                    py = img_init_polys
                    c_py = init['can_init_polys']
                    ct_01 = output['ct_01']

                py_ind = init['py_ind']
                pys = []
                # print(f"[DEBUG] py : {py.shape}")
                # if torch.numel(py) > 0:
                #     print(f"[DEBUG] py : {py.min()}, {py.max()}")

            if 'py_offset' in ret:
                pys_offset = ret['py_offset']
            else:
                pys_offset = []
            for i in range(self.iter):
                if self.pixel_norm_type == 'softmax':
                    pixel_map = F.softmax(output['pixel'][-1], dim=1)
                elif self.pixel_norm_type == 'trainable_softmax':
                    # 각 채널별 temperature 적용: exp(logit/temperature_c)
                    logits = output['pixel'][-1]  # [B, C, H, W]
                    # temperature를 양수로 보장 (softplus 또는 clamp 사용)
                    positive_temp = F.softplus(self.temperature) + 1e-8  # 최소값 보장
                    temp_reshaped = positive_temp.view(1, -1, 1, 1)
                    pixel_map = F.softmax(logits / temp_reshaped, dim=1)
                elif self.pixel_norm_type == 'trainable_softmax_softclamp':
                    # sigmoid를 사용한 soft clamping으로 temperature 범위 제한
                    logits = output['pixel'][-1]  # [B, C, H, W]
                    # T = T_lo + (T_hi - T_lo) * sigmoid(u)  ∈ [T_lo, T_hi]
                    T = self.T_lo + (self.T_hi - self.T_lo) * torch.sigmoid(self.u)  # [C]
                    temp_reshaped = T.view(1, -1, 1, 1)
                    pixel_map = F.softmax(logits / temp_reshaped, dim=1)
                elif self.pixel_norm_type == 'sep_trainable_sigmoid':
                    pixel_map = self.sep_sigmoid(output['pixel'][-1])
                elif self.pixel_norm_type == 'argmax':
                    pixel_map = torch.argmax(output['pixel'][-1], dim=1).unsqueeze(1)
                else:
                    pixel_map = output['pixel'][-1]

                # if pixel_map.size(3) != cnn_feature.size(3) or pixel_map.size(2) != cnn_feature.size(2):
                    # with torch.backends.cudnn.flags(enabled=True, deterministic=False):
                    # with torch.no_grad(): #edit to no_grad (for reprod)
                    # pixel_map = F.interpolate(pixel_map, size=(cnn_feature.size(2), cnn_feature.size(3)), mode='nearest')
                    # print(f"pixel_map : {pixel_map.shape}, cnn_feature : {cnn_feature.shape}")
                    # if self.training:
                    #     with torch.backends.cudnn.flags(enabled=True, deterministic=False):
                    #         pixel_map = F.interpolate(pixel_map, size=(cnn_feature.size(2), cnn_feature.size(3)),
                    #                                   mode='bilinear', align_corners=False).squeeze(1)
                    # else:
                    #     with torch.no_grad():
                    #         pixel_map = F.interpolate(pixel_map, size=(cnn_feature.size(2), cnn_feature.size(3)),
                    #                                   mode='bilinear', align_corners=False).squeeze(1)

                if self.projection is not None:
                    feature_combined = self.projection(torch.cat([pixel_map, cnn_feature], dim=1))
                else:
                    feature_combined = torch.cat([pixel_map, cnn_feature], dim=1)
                
                # ✅ weight sharing 여부에 따라 적절한 GCN 선택
                current_gcn = self.evolve_gcn if self.gcn_weight_sharing else self.evolve_gcn_list[i]
                
                if extract_offset:
                    if (self.use_vertex_classifier) and return_vertex_classifier:
                        py, py_isvalid, py_offset = self.evolve_poly(current_gcn, feature_combined, py, c_py, py_ind,
                                          stride=self.evolve_stride, extract_offset=True, ct_01=ct_01, return_vertex_classifier=True, cfg=self.cfg, iter_idx=i)
                        if py_isvalid is not None:
                            if 'py_valid_logits' in ret:
                                pre_py_valid_logits = ret.get('py_valid_logits', [])
                                pre_py_valid_logits.append(py_isvalid)
                                ret.update({'py_valid_logits': pre_py_valid_logits})
                            else:
                                ret.update({'py_valid_logits': [py_isvalid]})
                    else:
                        py, py_offset = self.evolve_poly(current_gcn, feature_combined, py, c_py, py_ind,
                                              stride=self.evolve_stride, extract_offset=True, ct_01=ct_01, cfg=self.cfg, iter_idx=i)
                    pys_offset.append(py_offset)
                else:
                    if self.use_vertex_classifier and return_vertex_classifier:
                        py, py_isvalid = self.evolve_poly(current_gcn, feature_combined, py, c_py, py_ind,
                                                          stride=self.evolve_stride, ct_01=ct_01,
                                                          return_vertex_classifier=True, cfg=self.cfg, iter_idx=i)
                        if py_isvalid is not None:
                            if 'py_valid_logits' in ret:
                                pre_py_valid_logits = ret.get('py_valid_logits', [])
                                pre_py_valid_logits.append(py_isvalid)
                                ret.update({'py_valid_logits': pre_py_valid_logits})
                            else:
                                ret.update({'py_valid_logits': [py_isvalid]})
                    else:
                        py = self.evolve_poly(current_gcn, feature_combined, py, c_py, py_ind,
                                              stride=self.evolve_stride, ct_01=ct_01, cfg=self.cfg, iter_idx=i)

                if not self.with_img_idx:
                    pys.append(py)
                elif ct_01.numel() > 0:
                    pys.append(py[ct_01])
                else:
                    pys.append(py)
                ret.setdefault('py_keys', []).append('deform')

                py = py / self.ro
                c_py = img_poly_to_can_poly(py)
            ret.update({'py': pys})
            ret.update({'py_offset': pys_offset})
            if 'batch_ind' not in ret:
                ret.update({'batch_ind': py_ind.to(device=py.device)})
            if 'ct_01' not in ret:
                ret.update({'ct_01': ct_01})

        # print(output.get('py_valid_logits', 'Impossible!'))
        return output

    def forward(self, output, cnn_feature, batch=None, test_stage='final', cfg=None, return_vertex_classifier=False, num_points_init=None):
        # ✅ cfg 업데이트 (학습/테스트 시 동적으로 전달받은 cfg 사용)
        if cfg is not None:
            self.cfg = cfg
            
        if batch is not None and 'test' not in batch['meta']:
            self.foward_train(output, batch, cnn_feature, return_vertex_classifier=return_vertex_classifier,
                              num_points_init=cfg.commen.init_points_per_poly if num_points_init is None else num_points_init)
        else:
            ignore = [False] * (self.iter + 1)
            if test_stage == 'coarse' or test_stage == 'init':
                ignore = [True for _ in ignore]
            if test_stage == 'final':
                ignore[-1] = True
            self.foward_test(output, cnn_feature, ignore=ignore, extract_offset=cfg.test.extract_offset if cfg is not None else False,
                             num_points_init=cfg.commen.init_points_per_poly if num_points_init is None else num_points_init, return_vertex_classifier=return_vertex_classifier)
        return output
