import torch.nn as nn
import torch
import torch.distributed as dist

def _assert_same_across_ranks(flag: bool, what: str, device): #edit:ddp-sync-bn-dummy:25-08-09
    if not dist.is_initialized():
        return
    t = torch.tensor([int(flag)], device=device)
    t_max = t.clone(); t_min = t.clone()
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
    if t_max.item() != t_min.item():
        rank = dist.get_rank()
        raise RuntimeError(f"[Rank {rank}] {what} mismatch across ranks")

class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, with_img_idx=False, refine_kernel_size=3):
        super(CircConv, self).__init__()
        self.with_img_idx = with_img_idx
        self.refine_kernel_size = refine_kernel_size
        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        if self.with_img_idx:
            self.fc = nn.Conv2d(state_dim, out_state_dim, kernel_size=(self.n_adj * 2 + 1, self.refine_kernel_size), dilation=(self.dilation, 1),
                                padding='same', padding_mode='circular')
        else:
            self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation, padding='same', padding_mode='circular')

    def forward(self, input):
        # if self.n_adj != 0:
        #     input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)

class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1, with_img_idx=False, refine_kernel_size=3):
        super(DilatedCircConv, self).__init__()
        self.with_img_idx = with_img_idx
        self.refine_kernel_size = refine_kernel_size
        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        if self.with_img_idx:
            self.fc = nn.Conv2d(state_dim, out_state_dim, kernel_size=(self.n_adj*2+1, self.refine_kernel_size), padding='same', padding_mode='circular', dilation=(self.dilation, 1))
        else:
            self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation)

    def forward(self, input):
        if self.n_adj != 0 and (not self.with_img_idx):
            input = torch.cat([input[..., -self.n_adj*self.dilation:], input, input[..., :self.n_adj*self.dilation]], dim=2)
        if input.size(-1) == 0:
            return self.fc(input.transpose(-1, 0))
        else:
            return self.fc(input)


# CircConv2D class removed - only using flatten mode now


_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv
}

class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1, with_img_idx=False, refine_kernel_size=3):
        super(BasicBlock, self).__init__()
        self.with_img_idx = with_img_idx
        if conv_type == 'grid':
            self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size)
        else:
            self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation, with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size)
        self.relu = nn.ReLU(inplace=True)
        if self.with_img_idx:
            self.norm = nn.BatchNorm2d(out_state_dim)
        else:
            self.norm = nn.BatchNorm1d(out_state_dim)

    @torch.no_grad()
    def _run_dummy_bn(self, ref_tensor: torch.Tensor):
        """빈 배치일 때 모든 rank에서 BN collective를 1회 맞춰주기 위한 더미 호출 (edit:ddp-sync-bn-dummy:25-08-09)"""
        C = self.norm.num_features
        # ref_tensor의 차원 수에 맞춰 채널 축만 맞는 최소 더미를 만듦
        dims = ref_tensor.dim()
        if self.with_img_idx:
            # BN2d: [N, C, H, W]
            dummy = torch.zeros(1, C, 1, 1, device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            # BN1d: [N, C] 또는 [N, C, L] 모두 OK → [1, C, 1]로 안전하게
            shape = (1, C, 1) if dims >= 3 else (1, C)
            dummy = torch.zeros(*shape, device=ref_tensor.device, dtype=ref_tensor.dtype)
        # training 모드에서 실행되어야 collective가 동작함 (eval은 collective 없음)
        was_training = self.norm.training
        self.norm.train(True)
        _ = self.norm(dummy)
        self.norm.train(was_training)

    def forward(self, x):
        # # 1) 입력 자체가 비었으면 conv/relu도 생략하고 더미 BN만 맞춘 뒤 그대로 반환 (edit:ddp-sync-bn-dummy:25-08-09)
        # if isinstance(x, torch.Tensor) and (x.numel() == 0 or x.shape[0] == 0):
        #     self._run_dummy_bn(x)
        #     return x  # 빈 텐서 그대로
        #
        # x = self.conv(x)
        # x = self.relu(x)
        # # 2) conv 결과가 비어도 동일 처리 (상류 모듈에서 필터링되면 여기서 비어질 수 있음) (edit:ddp-sync-bn-dummy:25-08-09)
        # if isinstance(x, torch.Tensor) and (x.numel() == 0 or x.shape[0] == 0):
        #     self._run_dummy_bn(x)
        #     return x
        #
        # # 3) 정상 경로
        # x = self.norm(x)
        # ---- 여기까지는 원래 경로 ----
        x = self.conv(x)
        x = self.relu(x)

        # ====== SyncBN 정합 강제 구간 (edit:ddp-sync-bn-dummy:25-08-09) ======
        is_syncbn = isinstance(self.norm, nn.SyncBatchNorm)

        # 1) BN의 training/eval 모드가 전 rank에서 같도록 보장
        #    (일부 분기에서 .eval() 된 상태가 섞이면 collective 불일치 발생)
        # _assert_same_across_ranks(self.norm.training, "SyncBatchNorm.training", x.device)

        # 2) AMP/dtype 정합: BN은 FP32로 고정해 호출 (rank별 autocast on/off 불일치 방지)
        if is_syncbn:
            with torch.cuda.amp.autocast(enabled=False):
                x = self.norm(x.float())
        else:
            x = self.norm(x)
        # ==================================
        return x


class BasicBlock2D(nn.Module):
    """BasicBlock for 2D inputs (first layer only)"""
    def __init__(self, state_dim, out_state_dim, n_adj=4, spatial_kernel=3):
        super(BasicBlock2D, self).__init__()
        # CircConv2D expects the input channel count, not the base feature dimension
        # It will internally flatten spatial dimensions from [C, H, W] to [C*H*W]
        self.conv = CircConv2D(state_dim, out_state_dim, n_adj=n_adj, spatial_kernel=spatial_kernel)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)  # 1D BatchNorm for output [N, C, V]

    @torch.no_grad()
    def _run_dummy_bn(self, ref_tensor: torch.Tensor):
        """빈 배치일 때 모든 rank에서 BN collective를 1회 맞춰주기 위한 더미 호출"""
        C = self.norm.num_features
        # BN1d: [1, C, 1] 형태로 더미 생성
        dummy = torch.zeros(1, C, 1, device=ref_tensor.device, dtype=ref_tensor.dtype)
        was_training = self.norm.training
        self.norm.train(True)
        _ = self.norm(dummy)
        self.norm.train(was_training)

    def forward(self, x):
        """
        x: [N_poly, C, N_vert, 3, 3] for 2D input
        output: [N_poly, out_dim, N_vert]
        """
        x = self.conv(x)  # [N_poly, out_dim, N_vert]
        x = self.relu(x)

        # SyncBN 정합 강제 구간 (기존과 동일)
        is_syncbn = isinstance(self.norm, nn.SyncBatchNorm)
        
        if is_syncbn:
            with torch.cuda.amp.autocast(enabled=False):
                x = self.norm(x.float())
        else:
            x = self.norm(x)
            
        return x


# class Snake(nn.Module):
#     def __init__(self, state_dim, feature_dim, conv_type='dgrid', with_img_idx=False, refine_kernel_size=3):
#         super(Snake, self).__init__()
#         self.with_img_idx = with_img_idx
#         self.head = BasicBlock(feature_dim, state_dim, conv_type, with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size)
#         self.res_layer_num = 7
#         dilation = [1, 1, 1, 2, 2, 4, 4]
#         n_adj = 4
#         for i in range(self.res_layer_num):
#             if dilation[i] == 0:
#                 conv_type = 'grid'
#             else:
#                 conv_type = 'dgrid'
#             conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=n_adj, dilation=dilation[i], with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size)
#             self.__setattr__('res'+str(i), conv)
#
#         fusion_state_dim = 256
#
#         if self.with_img_idx:
#             self.fusion = nn.Conv2d(state_dim * (self.res_layer_num + 1), fusion_state_dim, (1, refine_kernel_size), padding='same')
#             self.prediction = nn.Sequential(
#                 nn.Conv2d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, (1, refine_kernel_size), padding='same'),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(256, 64, (1, refine_kernel_size), padding='same'),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(64, 2, (1, refine_kernel_size), padding='same')
#             )
#         else:
#             self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, refine_kernel_size, padding=(refine_kernel_size-1)//2)
#             self.prediction = nn.Sequential(
#                 nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, refine_kernel_size, padding=(refine_kernel_size-1)//2),
#                 nn.ReLU(inplace=True),
#                 nn.Conv1d(256, 64, refine_kernel_size, padding=(refine_kernel_size-1)//2),
#                 nn.ReLU(inplace=True),
#                 nn.Conv1d(64, 2, refine_kernel_size, padding=(refine_kernel_size-1)//2)
#             )
#
#     def forward(self, x):
#         states = []
#         x = self.head(x)
#         states.append(x)
#         for i in range(self.res_layer_num):
#             x = self.__getattr__('res'+str(i))(x) + x
#             states.append(x)
#
#         state = torch.cat(states, dim=1)
#
#         global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
#         if self.with_img_idx:
#             global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2), global_state.size(3))
#         else:
#             global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
#         state = torch.cat([global_state, state], dim=1)
#
#         # print(f"[DEBUG] state.shape = {state.shape}, dtype={state.dtype}, device={state.device}")
#         x = self.prediction(state)
#
#         return x
class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid', with_img_idx=False, refine_kernel_size=3,
                 use_vertex_classifier=False, common_prediction_type=1, vtx_cls_kernel_size=1,
                 use_3x3_feature=False, feature_3x3_mode='flatten'):
        super(Snake, self).__init__()
        self.with_img_idx = with_img_idx
        self.use_vertex_classifier = use_vertex_classifier  # 👈 옵션 추가
        self.common_prediction_type = common_prediction_type
        self.use_3x3_feature = use_3x3_feature  # 3x3 feature 사용 여부
        self.feature_3x3_mode = feature_3x3_mode  # 'flatten' or 'conv2d'
        
        # Create head - feature_dim already includes flattening if needed
        # (get_gcn_feature_3x3 already flattens C*9 channels before passing here)
        head_feature_dim = feature_dim
        self.head = BasicBlock(head_feature_dim, state_dim, conv_type, with_img_idx=with_img_idx,
                               refine_kernel_size=refine_kernel_size)
        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        n_adj = 4
        for i in range(self.res_layer_num):
            conv_type_i = 'grid' if dilation[i] == 0 else 'dgrid'
            conv = BasicBlock(state_dim, state_dim, conv_type_i, n_adj=n_adj, dilation=dilation[i],
                              with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size)
            self.__setattr__('res' + str(i), conv)

        fusion_state_dim = 256

        if self.with_img_idx:
            self.fusion = nn.Conv2d(state_dim * (self.res_layer_num + 1), fusion_state_dim, (1, refine_kernel_size),
                                    padding='same')
            self.prediction = nn.Sequential(
                nn.Conv2d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, (1, refine_kernel_size),
                          padding='same'),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 64, (1, refine_kernel_size), padding='same'),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, (1, refine_kernel_size), padding='same')
            )
        else:
            self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, refine_kernel_size,
                                    padding=(refine_kernel_size - 1) // 2)

            # Calculate input channels for prediction layers
            prediction_input_dim = state_dim * (self.res_layer_num + 1) + fusion_state_dim
            
            if self.use_vertex_classifier:
                if self.common_prediction_type == 1:
                    self.common_prediction = nn.Sequential(
                        nn.Conv1d(prediction_input_dim, 256, refine_kernel_size,
                                  padding=(refine_kernel_size - 1) // 2),
                        nn.ReLU(inplace=True)
                    )
                    self.offset_prediction = nn.Sequential(
                        nn.Conv1d(256, 64, refine_kernel_size, padding=(refine_kernel_size - 1) // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(64, 2, refine_kernel_size, padding=(refine_kernel_size - 1) // 2)
                    )
                    self.vertex_classifier = nn.Sequential(
                        nn.Conv1d(256, 64, kernel_size=vtx_cls_kernel_size, padding='same'),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(64, 2, kernel_size=vtx_cls_kernel_size, padding='same')
                    )

                elif self.common_prediction_type == 2:
                    self.common_prediction = nn.Sequential(
                        nn.Conv1d(prediction_input_dim, 256, refine_kernel_size,
                                  padding=(refine_kernel_size - 1) // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(256, 64, refine_kernel_size, padding=(refine_kernel_size - 1) // 2),
                        nn.ReLU(inplace=True)
                    )
                    self.offset_prediction = nn.Sequential(
                        nn.Conv1d(64, 2, refine_kernel_size, padding=(refine_kernel_size - 1) // 2)
                    )
                    self.vertex_classifier = nn.Sequential(
                        nn.Conv1d(64, 2, kernel_size=vtx_cls_kernel_size, padding='same')
                    )

                elif self.common_prediction_type == 0:
                    self.offset_prediction = nn.Sequential(
                        nn.Conv1d(prediction_input_dim, 256, refine_kernel_size,
                                  padding=(refine_kernel_size - 1) // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(256, 64, refine_kernel_size, padding=(refine_kernel_size - 1) // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(64, 2, refine_kernel_size, padding=(refine_kernel_size - 1) // 2)
                    )
                    self.vertex_classifier = nn.Sequential(
                        nn.Conv1d(prediction_input_dim, 256, kernel_size=vtx_cls_kernel_size, padding='same'),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(256, 64, kernel_size=vtx_cls_kernel_size, padding='same'),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(64, 2, kernel_size=vtx_cls_kernel_size, padding='same')
                    )
            else:
                self.prediction = nn.Sequential(
                    nn.Conv1d(prediction_input_dim, 256, refine_kernel_size,
                              padding=(refine_kernel_size - 1) // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(256, 64, refine_kernel_size, padding=(refine_kernel_size - 1) // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(64, 2, refine_kernel_size, padding=(refine_kernel_size - 1) // 2)
                )

    def forward(self, x, return_vertex_classifier=False):
        states = []
        coord_part = None  # Initialize coordinate part
        
        # Standard processing for all modes
        x = self.head(x)
        
        # Process through residual layers with clean state_dim channels
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res' + str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)

        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        if self.with_img_idx:
            global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2),
                                               global_state.size(3))
        else:
            global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)
        
        # No coordinate concatenation needed in simplified version

        if self.use_vertex_classifier:
            if self.common_prediction_type == 0:
                offsets = self.offset_prediction(state)
                logits = self.vertex_classifier(state)
            else:
                shared = self.common_prediction(state)
                offsets = self.offset_prediction(shared)
                logits = self.vertex_classifier(shared)

            if return_vertex_classifier:
                return offsets, logits  # x: offset prediction, logits: (B, 1, N)
            else:
                return offsets
        else:
            x = self.prediction(state)  # shape: (B, 2, N)
            return x

    #edit:ddp-sync-bn-dummy:25-08-09
    @torch.no_grad()
    def run_dummy_like(self, ref: torch.Tensor):
        # 항상 FP32로 BN 호출 (AMP 불일치 방지)
        with torch.cuda.amp.autocast(enabled=False):
            device = ref.device
            for m in self.modules():
                if isinstance(m, nn.SyncBatchNorm):
                    C = m.num_features
                    was_training = m.training
                    m.train(True)
                    ok = False
                    for shape in [(1, C, 1, 1), (1, C, 1), (1, C)]:
                        try:
                            dummy = torch.zeros(*shape, device=device, dtype=torch.float32)
                            m(dummy)
                            ok = True
                            break
                        except Exception:
                            pass
                    if not ok:
                        # 마지막 안전장치: 그래도 한 번은 호출되게
                        dummy = torch.zeros(1, C, device=device, dtype=torch.float32)
                        m(dummy)
                    m.train(was_training)


class AggregateCirConv(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid', with_img_idx=False, refine_kernel_size=3, type='default',
                 fusion_conv_num=3, fusion_state_dim=256, refine_kernel_size_fusion=1):
        super(AggregateCirConv, self).__init__()
        self.aggregate_type = type #(default, snake)
        self.fusion_conv_num = fusion_conv_num
        self.fusion_state_dim = fusion_state_dim
        self.with_img_idx = with_img_idx
        self.head = BasicBlock(feature_dim, state_dim, conv_type, with_img_idx=with_img_idx,
                               refine_kernel_size=refine_kernel_size)
        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        n_adj = 4
        for i in range(self.res_layer_num):
            if dilation[i] == 0:
                conv_type = 'grid'
            else:
                conv_type = 'dgrid'
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=n_adj, dilation=dilation[i],
                              with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size)
            self.__setattr__('res' + str(i), conv)

        if self.aggregate_type == 'snake':
            self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, refine_kernel_size,
                                    padding=(refine_kernel_size - 1) // 2)
            self.prediction = nn.Sequential(
                nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, refine_kernel_size,
                          padding=(refine_kernel_size - 1) // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 64, refine_kernel_size, padding=(refine_kernel_size - 1) // 2),
            )
        else:
            self.fusion = nn.Sequential()
            pre_dim = state_dim * (self.res_layer_num + 1)
            for fusion_i in range(self.fusion_conv_num):
                if fusion_i == (self.fusion_conv_num-1):
                    self.fusion.add_module(
                        f"conv1d_{fusion_i}", nn.Conv1d(pre_dim, 64, refine_kernel_size_fusion)
                    )
                    self.fusion.add_module(
                        f"relu_{fusion_i}", nn.ReLU(inplace=True)
                    )
                    continue
                self.fusion.add_module(
                    f"conv1d_{fusion_i}", nn.Conv1d(pre_dim, fusion_state_dim, refine_kernel_size_fusion)
                )
                self.fusion.add_module(
                    f"relu_{fusion_i}", nn.ReLU(inplace=True)
                )
                pre_dim = fusion_state_dim

    def forward(self, x):
        states = []
        x = self.head(x)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res' + str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)

        if self.aggregate_type == 'snake':
            global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
            global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
            state = torch.cat([global_state, state], dim=1)
            x = self.prediction(state)
        else:
            x = self.fusion(state)
        return x
