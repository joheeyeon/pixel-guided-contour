import torch.nn as nn
import torch
from .utils import img_poly_to_can_poly, get_gcn_feature
class IPC(nn.Module):
    def __init__(self, input_dim=18, num_params=[8,8,1], ro=4., random_sample_range=5., match_dist_p=2.0, dynamic=True):
        super(IPC, self).__init__()
        self.match_dist_p = match_dist_p
        self.random_sample_range = random_sample_range
        self.ro = ro
        self.input_dim = input_dim
        self.num_params = num_params
        self.dynamic = dynamic
        self.net = nn.Sequential(
                        nn.Conv1d(self.input_dim, num_params[0],
                                  kernel_size=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(num_params[0], num_params[1],
                                  kernel_size=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(num_params[1], num_params[2],
                                  kernel_size=1, bias=True))
        # freeze network
        if self.dynamic:
            for param in self.net.parameters():
                param.requires_grad = False

    def get_angle_bw_vectors(self, vector_1, vector_2):
        unit_vector_1 = vector_1 / torch.norm(vector_1, dim=-1, keepdim=True).expand(-1,-1,2)
        unit_vector_2 = vector_2 / torch.norm(vector_2, dim=-1, keepdim=True).expand(-1,-1,2)
        dot_product = torch.mul(unit_vector_1, unit_vector_2).sum(-1)
        angle = torch.acos(dot_product)
        return angle

    def to_unit(self, vec):
        return vec / torch.norm(vec, dim=-1, keepdim=True).expand(-1, -1, vec.size(-1))

    def cross_prod(self, vec1, vec2):
        return vec1[..., 0] * vec2[..., 1] - vec1[..., 1] * vec2[..., 0]

    def get_ipc_gt(self, gt_py, pred_py, len_keys=None):
        # re-sort gt (matching with min distance)
        mat_dist = torch.cdist(pred_py, gt_py, p=self.match_dist_p)
        if len_keys is not None:
            for b in range(len_keys.size(0)):
                mat_dist[b, :, len_keys[b]:] = float('inf')

        min_dist_ids = mat_dist.argmin(-1, keepdim=True).expand(-1, -1, 2)
        matched_gt = torch.gather(gt_py, 1, min_dist_ids)
        if len_keys is not None:
            matched_gt_next = torch.gather(gt_py, 1, torch.remainder(min_dist_ids + 1, len_keys.unsqueeze(-1).unsqueeze(-1).expand(-1, min_dist_ids.size(1), min_dist_ids.size(2))))
            matched_gt_pre = torch.gather(gt_py, 1, torch.remainder(min_dist_ids - 1, len_keys.unsqueeze(-1).unsqueeze(-1).expand(-1, min_dist_ids.size(1), min_dist_ids.size(2))))
        else:
            matched_gt_next = torch.gather(gt_py, 1, torch.remainder(min_dist_ids + 1, gt_py.size(1)))
            matched_gt_pre = torch.gather(gt_py, 1, torch.remainder(min_dist_ids - 1, gt_py.size(1)))

        # get each vectors
        vec_to_next_compact = self.to_unit(matched_gt_next - matched_gt)
        vec_to_pre_compact = self.to_unit(matched_gt_pre - matched_gt)
        vec_to_pred_compact = self.to_unit(pred_py - matched_gt)
        # calculate cross products
        cross_next_to_pre = torch.round(1.5*self.cross_prod(vec_to_next_compact, vec_to_pre_compact))
        cross_next_to_pred = torch.round(1.5*self.cross_prod(vec_to_next_compact, vec_to_pred_compact))
        cross_pred_to_pre = torch.round(1.5*self.cross_prod(vec_to_pred_compact, vec_to_pre_compact))
        # get gt for each case, acute and obtuse
        mask_acute = torch.logical_and(cross_next_to_pre >= 0,
                                       torch.logical_and(cross_next_to_pred >= 0, cross_pred_to_pre >= 0))
        mask_obtuse = torch.logical_and(cross_next_to_pre < 0,
                                        torch.logical_or(cross_next_to_pred >= 0, cross_pred_to_pre >= 0))
        # get combined mask & gt
        mask_border = torch.logical_or(torch.logical_or(cross_next_to_pred == 0, cross_pred_to_pre == 0),
                                     torch.logical_or(cross_next_to_pred.isnan(),cross_pred_to_pre.isnan()))
        # mask_with_dist = torch.round(torch.sum(torch.sqrt((matched_gt - pred_py).pow(2)), -1)) < 3
        mask_with_dist = torch.round(mat_dist.min(-1)[0]) < 2
        mask_border = torch.logical_or(mask_border, mask_with_dist)
        gt_ipc = torch.logical_or(mask_acute, mask_obtuse).float()
        gt_ipc[mask_border] = 0.5
        return gt_ipc, mask_border, matched_gt

    def forward(self, output, fine_grained_feature, batch=None, train_gt=False, cut_py_grad=False, cut_feature_grad=False):
        h, w = fine_grained_feature.size(2), fine_grained_feature.size(3)
        if train_gt and self.training:
            # random sampling round img_gt_polys
            rand_sign = torch.rand(output['img_gt_polys'].size(), device=output['img_gt_polys'].device)
            rand_sign = torch.where(rand_sign >= 0.5, 1., -1.).float()
            rand_offset = rand_sign * self.random_sample_range * torch.rand(output['img_gt_polys'].size(), device=output['img_gt_polys'].device)
            py_pred = (output['img_gt_polys'] + rand_offset) / self.ro
        else:
            py_pred = output['py_pred' if 'py_pred' in output else 'py'][-1] / self.ro

        if cut_py_grad:
            py_pred = py_pred.clone()

        vertex_feature = get_gcn_feature(fine_grained_feature, py_pred, output['batch_ind'], h, w)
        vertex_params = get_gcn_feature(output['bound_control'], output['ct'].unsqueeze(1), output['batch_ind'], h, w)

        c_py_pred = img_poly_to_can_poly(py_pred)
        c_py_pred = c_py_pred * self.ro
        if cut_feature_grad:
            vertex_feature = vertex_feature.clone()
        ipc_input = torch.cat([vertex_feature, c_py_pred.permute(0, 2, 1)], dim=1)
        ipc_output = torch.zeros([ipc_input.size(0),ipc_input.size(-1)], device=ipc_input.device, dtype=torch.float32)
        if self.dynamic:
            # load conv weight
            for n in range(ipc_input.size(0)):
                pre_num = self.input_dim
                num_stack_params = 0
                for i in range(len(self.num_params)):
                    # print(f"{i} : {vertex_params[n,num_stack_params:num_stack_params+pre_num*self.num_params[i],:].shape}")
                    weights = vertex_params[n,num_stack_params:num_stack_params+pre_num*self.num_params[i],:].contiguous().view(self.num_params[i],pre_num,1)
                    bias = vertex_params[n,num_stack_params+pre_num*self.num_params[i]:num_stack_params+pre_num*self.num_params[i]+self.num_params[i],0]
                    self.net[2*i].weight = nn.Parameter(weights, requires_grad=False)
                    self.net[2*i].bias = nn.Parameter(bias, requires_grad=False)
                    num_stack_params = num_stack_params+pre_num*self.num_params[i]+self.num_params[i]
                    pre_num = self.num_params[i]

                ipc_output[n] = self.net(ipc_input[n].unsqueeze(0))
        else:
            for n in range(ipc_input.size(0)):
                ipc_output[n] = self.net(ipc_input[n].unsqueeze(0))

        if self.training:
            # gt_py_key, len_keys = get_keypoints_for_normvec(output['img_gt_polys'])
            ipc_gt, ipc_mask_border, matched_gt = self.get_ipc_gt(output['img_gt_polys'].clone(), py_pred * self.ro)
            # ''' tmp '''
            # if 'ipc_gt' in output:
            #     print(len(output['ipc_gt']))
            # else:
            #     print(f"first")
            #
            # import numpy as np
            # import matplotlib.pyplot as plt
            # mean = np.array([0.40789654, 0.44719302, 0.47026115],
            #                 dtype=np.float32).reshape(1, 1, 3)
            # std = np.array([0.28863828, 0.27408164, 0.27809835],
            #                dtype=np.float32).reshape(1, 1, 3)
            # gt = output['img_gt_polys']
            # gt = gt.clone().cpu().numpy()
            # match_gt = matched_gt.clone().detach().cpu().numpy()
            # ex = (py_pred * self.ro).clone().detach().cpu().numpy()
            # mask_1 = (ipc_gt == 1).clone().detach().cpu().numpy()
            # mask_0 = (ipc_gt == 0).clone().detach().cpu().numpy()
            # mask_border = (ipc_gt == 0.5).clone().detach().cpu().numpy()
            # for b in range(batch['inp'].size(0)):
            #     inp = bgr_to_rgb(unnormalize_img(batch['inp'][b], mean, std).permute(1, 2, 0))
            #     idx_b = np.where(b == output['batch_ind'].clone().detach().cpu().numpy())
            #
            #     fig, ax = plt.subplots(1, figsize=(20, 10))
            #     fig.tight_layout()
            #     ax.axis('off')
            #     ax.imshow(inp)
            #
            #     for i in range(len(ex[idx_b])):
            #         points = ex[idx_b][i]
            #         poly = ex[idx_b][i]
            #         poly_gt = gt[idx_b][i]
            #         points_gt = gt[idx_b][i]
            #         poly = np.append(poly, [poly[0]], axis=0)
            #         poly_gt = np.append(poly_gt, [poly_gt[0]], axis=0)
            #         ax.plot(poly[:, 0], poly[:, 1], color='b', lw=2)
            #         ax.plot(poly_gt[:, 0], poly_gt[:, 1], color='g', lw=2)
            #         ax.plot(points_gt[:, 0], points_gt[:,1], 'go')
            #         # print(f"mask_1 : {mask_1.shape}, points : {points.shape}")
            #         ax.plot(points[mask_0[idx_b][i], 0], points[mask_0[idx_b][i], 1], 'ro')
            #         ax.plot(points[mask_1[idx_b][i], 0], points[mask_1[idx_b][i], 1], 'yo')
            #         ax.plot(points[mask_border[idx_b][i], 0], points[mask_border[idx_b][i], 1], 'w*')
            #         for match_i in range(match_gt.shape[1]):
            #             ax.plot(np.array([match_gt[idx_b][i][match_i,0],points[match_i,0]]), np.array([match_gt[idx_b][i][match_i,1],points[match_i,1]]), color='m')
            #
            #     plt.show()
            # ''' tmp end '''
            if train_gt:
                if 'ipc_gt_random' in output:
                    output['ipc_gt_random'].append(ipc_gt.float())
                    output['ipc_mask_border_random'].append(ipc_mask_border)
                    output['ipc_train_py_random'].append(py_pred * self.ro)
                else:
                    output.update({f'ipc_gt_random': [ipc_gt.float()], f'ipc_mask_border_random': [ipc_mask_border],
                                   'ipc_train_py_random': [py_pred * self.ro]})
            else:
                if 'ipc_gt' in output:
                    output['ipc_gt'].append(ipc_gt.float())
                    output['ipc_mask_border'].append(ipc_mask_border)
                    output['ipc_train_py'].append(py_pred * self.ro)
                else:
                    output.update({f'ipc_gt': [ipc_gt.float()], f'ipc_mask_border': [ipc_mask_border], 'ipc_train_py': [py_pred * self.ro]})

        if train_gt:
            if 'ipc_random' in output:
                output['ipc_random'].append(ipc_output)
            else:
                output.update({f'ipc_random': [ipc_output]})
        else:
            if 'ipc' in output:
                output['ipc'].append(ipc_output)
            else:
                output.update({f'ipc': [ipc_output]})
        return output