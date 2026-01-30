import torch
from .utils import decode_ct_hm, clip_to_image, get_gcn_feature, uniform_upsample, global_to_local_ct_img_idx

class Refine(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, stride=4., num_point_each_step=None, with_img_idx=False, kernel_size=3, use_trans_feature=True, cat_include_coarse=False):
        super(Refine, self).__init__()
        self.num_point = num_point
        self.stride = stride
        self.num_point_each_step = num_point_each_step
        self.with_img_idx = with_img_idx
        if use_trans_feature:
            # When cat_include_coarse is True, pixel head output (2 channels) is concatenated
            input_channels = c_in + 2 if cat_include_coarse else c_in
            self.trans_feature = torch.nn.Sequential(torch.nn.Conv2d(input_channels, 256, kernel_size=3,
                                                                     padding=1, bias=True),
                                                     torch.nn.ReLU(inplace=True),
                                                     torch.nn.Conv2d(256, 64, kernel_size=1,
                                                                     stride=1, padding=0, bias=True))
        else:
            self.trans_feature = None
        if self.with_img_idx:
            self.trans_poly = torch.nn.Conv1d(kernel_size=kernel_size, in_channels=((num_point + 1) * 64),
                                              out_channels=num_point * 4, bias=False, padding='same')
            self.trans_fuse = torch.nn.Conv1d(kernel_size=kernel_size, in_channels=num_point * 4,
                                              out_channels=num_point * 2, bias=True, padding='same')
        else:
            self.trans_poly = torch.nn.Linear(in_features=((num_point + 1) * 64),
                                              out_features=num_point * 4, bias=False)
            self.trans_fuse = torch.nn.Linear(in_features=num_point * 4,
                                              out_features=num_point * 2, bias=True)
            if self.num_point_each_step is not None:
                for i_step, n_pts in enumerate(self.num_point_each_step):
                    setattr(self, f'trans_poly{i_step}', torch.nn.Linear(in_features=((n_pts + 1) * 64),
                                              out_features=n_pts * 4, bias=False))
                    setattr(self, f'trans_fuse{i_step}', torch.nn.Linear(in_features=n_pts * 4,
                                              out_features=n_pts * 2, bias=True))

    def global_deform(self, points_features, init_polys, additional_step=None):
        if additional_step is None:
            points_features = self.trans_poly(points_features)
            offsets = self.trans_fuse(points_features)
        else:
            points_features = getattr(self,f'trans_poly{additional_step}')(points_features)
            offsets = getattr(self,f'trans_fuse{additional_step}')(points_features)
        if self.with_img_idx:
            offsets = offsets.transpose(1, 2)
            offsets = offsets.view(offsets.size(0), offsets.size(1), self.num_point, 2)
        else:
            poly_num = init_polys.size(0)
            if additional_step is None:
                offsets = offsets.view(poly_num, self.num_point, 2)
            else:
                offsets = offsets.view(poly_num, self.num_point_each_step[additional_step], 2)
        coarse_polys = offsets * self.stride + init_polys.detach()
        return coarse_polys

    def forward(self, feature, ct_polys, init_polys, ct_img_idx, ignore=False, ct_01=None, get_feature=False):
        if ignore or len(init_polys) == 0:
            return init_polys, feature
        h, w = feature.size(2), feature.size(3)
        poly_num = ct_polys.size(0)
    
        if self.trans_feature is not None:
            feature = self.trans_feature(feature) # feature : dim redcution

        ct_polys = ct_polys.unsqueeze(1).expand(init_polys.size(0), 1, init_polys.size(2)) # for with_imd_idx=False; (N_py, 2) > (N_py, 1, 2)
        pre_polys = init_polys #(N_py, N_vert, 2)
        if self.num_point_each_step is not None:
            for i_step, n_pts in enumerate(self.num_point_each_step):
                pre_polys = uniform_upsample(pre_polys.unsqueeze(0), n_pts).squeeze(0)
                points = torch.cat([ct_polys, pre_polys], dim=1)
                feature_points = get_gcn_feature(feature, points, ct_img_idx, h, w).view(poly_num, -1)
                coarse_polys = self.global_deform(feature_points, pre_polys, additional_step=i_step)
                pre_polys = coarse_polys

            pre_polys = uniform_upsample(pre_polys.unsqueeze(0), self.num_point).squeeze(0)
        points = torch.cat([ct_polys, pre_polys], dim=1)#(N_py, N_vert+1, 2)
        # print(f"ct_img_idx : {ct_img_idx.shape}") -> [100]

        feature_points = (get_gcn_feature(feature, points, ct_img_idx, h, w)).view(poly_num, -1) #(N_py, dim_feature=64, N_vert+1)
        # raise KeyboardInterrupt
        # debug_tensor("feature_points", feature_points) #-1~1
        if self.with_img_idx:
            feature_points_reshape = torch.zeros([feature.size(0), ct_01.size(1), feature_points.size(-1)]).to(feature_points.device)
            feature_points_reshape[ct_01] = feature_points
            feature_points_reshape = feature_points_reshape.transpose(1, 2)
        else:
            feature_points_reshape =  feature_points

        if self.with_img_idx:
            pre_polys_reshape = torch.zeros([feature.size(0), ct_01.size(1), pre_polys.size(1), pre_polys.size(2)]).to(
                pre_polys.device)
            pre_polys_reshape[ct_01] = pre_polys
        else:
            pre_polys_reshape = pre_polys
        coarse_polys = self.global_deform(feature_points_reshape, pre_polys_reshape)
        return coarse_polys, feature

# def debug_tensor(name, tensor):
#     if torch.isnan(tensor).any():
#         print(f"[DEBUG] NaN found in {name}")
#     if torch.isinf(tensor).any():
#         print(f"[DEBUG] Inf found in {name}")
#     if tensor.numel() > 0:
#         print(f"[DEBUG] {name}: min={tensor.min().item()}, max={tensor.max().item()}")
#     else:
#         print(f"[DEBUG] {name} is EMPTY")

class Decode(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, init_stride=10., coarse_stride=4., down_sample=4., min_ct_score=0.05, num_point_each_step=None,
                 with_img_idx=False, refine_kernel_size=3, use_dp=True, use_trans_feature=True, cat_include_coarse=False):
        super(Decode, self).__init__()
        self.use_dp = use_dp
        self.stride = init_stride
        self.down_sample = down_sample
        self.min_ct_score = min_ct_score
        self.with_img_idx = with_img_idx
        self.refine = Refine(c_in=c_in, num_point=num_point, stride=coarse_stride, num_point_each_step=num_point_each_step,
                             with_img_idx=with_img_idx, kernel_size=refine_kernel_size, use_trans_feature=use_trans_feature, cat_include_coarse=cat_include_coarse)

    def train_decode(self, data_input, output, cnn_feature, get_feature=False):
        wh_pred = output['wh']
        # print(f"wh_pred : {wh_pred.shape}")
        ct_01 = data_input['ct_01'].bool()
        ct_ind = data_input['ct_ind'][ct_01]
        ct_img_idx = data_input['ct_img_idx'][ct_01]
        # ✅ 로컬 index 변환
        if (torch.cuda.device_count() > 1) and self.use_dp:
            device_id = cnn_feature.device.index
            world_size = torch.cuda.device_count()
            total_batch_size = data_input['inp'].size(0)
            ct_img_idx = global_to_local_ct_img_idx(ct_img_idx, device_id, world_size, total_batch_size)

        # 원래 ct_hm 그리드에서의 (x,y)
        _, _, ct_H, ct_W = data_input['ct_hm'].size()
        ct_x = ct_ind % ct_W
        ct_y = torch.div(ct_ind, ct_W, rounding_mode='trunc')
        # wh_pred 해상도
        _, _, wh_H, wh_W = wh_pred.size()

        #======== edit:ccp+ive-ct_xy:25-08-09 ============
        if (wh_H != ct_H) or (wh_W != ct_W):
            # ---- 서로 다른 그리드 간 스케일 보정 ----
            scale_x = float(wh_W) / float(ct_W)
            scale_y = float(wh_H) / float(ct_H)

            # 인덱싱용 (정수, round + clamp)
            ct_x_idx = torch.clamp((ct_x.to(torch.float32) * scale_x).round().to(torch.long), 0, wh_W - 1)
            ct_y_idx = torch.clamp((ct_y.to(torch.float32) * scale_y).round().to(torch.long), 0, wh_H - 1)

            # 중심 좌표합산용 (float)
            ct_x_ctr = ct_x.to(torch.float32) * scale_x
            ct_y_ctr = ct_y.to(torch.float32) * scale_y
        else:
            ct_x_idx = ct_x
            ct_y_idx = ct_y
            ct_x_ctr = ct_x.to(torch.float32)
            ct_y_ctr = ct_y.to(torch.float32)

        # gather offsets
        # wh_pred: (N, C=2*K, H, W) → (M, 2, K)
        if ct_x_idx.size(0) == 0:
            ct_offset = wh_pred[ct_img_idx, :, ct_y_idx, ct_x_idx].view(
                0, torch.div(wh_pred.size(1), 2, rounding_mode='trunc'), 2
            )
        else:
            ct_offset = wh_pred[ct_img_idx, :, ct_y_idx, ct_x_idx].view(ct_x_idx.size(0), -1, 2)

        # 중심 좌표 (wh_pred 그리드 기준)
        ct = torch.stack([ct_x_ctr, ct_y_ctr], dim=1)  # (M,2) float

        # 초기 폴리곤: offset * stride + center
        init_polys = ct_offset * self.stride + ct.unsqueeze(1).expand(ct_offset.size(0),
                                                                      ct_offset.size(1), ct_offset.size(2))

        # refine
        # coarse_polys, feature_coarse = self.refine(cnn_feature, ct, init_polys, ct_img_idx.clone(), ct_01=ct_01, get_feature=get_feature)
        coarse_polys, feature_coarse = self.refine(cnn_feature, ct, init_polys, ct_img_idx, ct_01=ct_01,
                                                   get_feature=get_feature)
        # debug_tensor("coarse_polys", coarse_polys) #0~511
        if self.with_img_idx:
            init_polys_reshape = torch.zeros([ct_01.size(0), ct_01.size(1), init_polys.size(1), init_polys.size(2)]).to(init_polys.device)
            init_polys_reshape[ct_01] = init_polys
        else:
            init_polys_reshape = init_polys
        output.update({'poly_init': init_polys_reshape * self.down_sample})
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        return feature_coarse, output

    def test_decode(self, cnn_feature, output, K=100, min_ct_score=0.05, ignore_gloabal_deform=False, get_feature=False):
        # hm_pred, wh_pred = output['ct_hm'], output['wh']
        # poly_init, detection, wh_pred = decode_ct_hm(torch.sigmoid(hm_pred), wh_pred,
        #                                     K=K, stride=self.stride)
        # valid = detection[..., 2] >= min_ct_score
        # poly_init, detection, wh_pred = poly_init[0][valid], detection[0][valid], wh_pred[0][valid]
        #
        # init_polys = clip_to_image(poly_init, cnn_feature.size(2), cnn_feature.size(3))
        # output.update({'poly_init': init_polys * self.down_sample})
        #
        # img_id = torch.zeros((len(poly_init), ), dtype=torch.int64)
        # ct_01 = torch.ones([cnn_feature.size(0), len(poly_init)], dtype=torch.bool)
        # poly_coarse, feature_coarse = self.refine(cnn_feature, detection[:, :2], poly_init, img_id, ignore=ignore_gloabal_deform, ct_01=ct_01, get_feature=get_feature)
        # coarse_polys = clip_to_image(poly_coarse, cnn_feature.size(2), cnn_feature.size(3))
        # output.update({'poly_coarse': coarse_polys * self.down_sample})
        # output.update({'detection': detection, 'wh_pred': wh_pred, 'ct_01': ct_01.to(coarse_polys.device)})
        # output.update({'py_ind': img_id.to(torch.int32)})
        hm_pred, wh_pred = output['ct_hm'], output['wh']  # [B, C, H, W], [B, 2, H, W]
        B, _, H, W = hm_pred.shape

        poly_init, detection, wh_pred = decode_ct_hm(
            torch.sigmoid(hm_pred), wh_pred, K=K, stride=self.stride
        )  # poly_init: [B, K, V, 2], detection: [B, K, 5], wh_pred: [B, K, 2]

        # flatten across batch
        B, K, *_ = detection.shape
        poly_init = poly_init.view(B * K, *poly_init.shape[2:])  # [B*K, V, 2]
        detection = detection.view(B * K, -1)  # [B*K, D]
        wh_pred = wh_pred.view(B * K, -1)  # [B*K, 2]

        # compute original batch index for each polygon
        img_id = torch.arange(B, device=hm_pred.device).repeat_interleave(K)  # [B*K]

        # filter by score
        score_mask = detection[:, 2] >= min_ct_score  # [B*K]
        poly_init = poly_init[score_mask]
        detection = detection[score_mask]
        wh_pred = wh_pred[score_mask]
        img_id = img_id[score_mask]  # [N_valid]

        # ct_01: [B, N_valid], one-hot where ct_01[b, i] = True if polygon i is from batch b
        N_valid = img_id.shape[0]
        ct_01 = torch.zeros(B, N_valid, dtype=torch.bool, device=img_id.device)
        ct_01[img_id, torch.arange(N_valid, device=img_id.device)] = True

        init_polys = clip_to_image(poly_init, H, W)
        output.update({'poly_init': init_polys * self.down_sample})
        output.update({'detection': detection, 'wh_pred': wh_pred})
        output.update({'ct_01': ct_01})
        output.update({'py_ind': img_id.to(torch.int32)})

        poly_coarse, feature_coarse = self.refine(
            cnn_feature, detection[:, :2], poly_init, img_id,
            ignore=ignore_gloabal_deform, ct_01=ct_01, get_feature=get_feature
        )

        coarse_polys = clip_to_image(poly_coarse, H, W)
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        return feature_coarse, output

    def forward(self, data_input, cnn_feature, output=None, is_training=True, ignore_gloabal_deform=False, get_feature=False):
        if is_training:
            feature_coarse, output = self.train_decode(data_input, output, cnn_feature, get_feature=get_feature)
        else:
            feature_coarse, output = self.test_decode(cnn_feature, output, min_ct_score=self.min_ct_score,
                             ignore_gloabal_deform=ignore_gloabal_deform, get_feature=get_feature)
        return feature_coarse, output

