import torch.nn as nn
from .snake import Snake
from .utils import prepare_training, prepare_testing_init, img_poly_to_can_poly, get_gcn_feature
import torch


class Evolution(nn.Module):
    def __init__(self, evole_ietr_num=3, evolve_stride=1., ro=4., with_img_idx=False, refine_kernel_size=3, in_featrue_dim=64,
                 use_vertex_classifier=False, num_vertex=128):
        super(Evolution, self).__init__()
        assert evole_ietr_num >= 1
        self.use_vertex_classifier = use_vertex_classifier
        self.evolve_stride = evolve_stride
        self.ro = ro
        self.num_vertex = num_vertex
        self.evolve_gcn = Snake(state_dim=self.num_vertex, feature_dim=in_featrue_dim+2, conv_type='dgrid', with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size)
        self.iter = evole_ietr_num - 1
        self.with_img_idx = with_img_idx

        for i in range(self.iter):
            evolve_gcn = Snake(state_dim=self.num_vertex, feature_dim=in_featrue_dim+2, conv_type='dgrid', with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size, use_vertex_classifier=self.use_vertex_classifier if i == (self.iter-1) else False)
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)

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
        init = prepare_testing_init(polys, self.ro, num_points=num_points_init, py_ind=output['py_ind'])
        return init

    def prepare_testing_evolve(self, output, h, w):
        img_init_polys = output['img_init_polys']
        img_init_polys[..., 0] = torch.clamp(img_init_polys[..., 0], min=0, max=w-1)
        img_init_polys[..., 1] = torch.clamp(img_init_polys[..., 1], min=0, max=h-1)
        output.update({'img_init_polys': img_init_polys})
        return img_init_polys
    
    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, stride=1., ignore=False, extract_offset=False, ct_01=None, return_vertex_classifier=False):
        if ignore:
            if extract_offset:
                return i_it_poly * self.ro, i_it_poly * 0.0
            else:
                return i_it_poly * self.ro
        if len(i_it_poly) == 0:
            if extract_offset:
                return torch.zeros_like(i_it_poly), torch.zeros_like(i_it_poly)
            else:
                return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        if self.with_img_idx:
            i_it_poly_compact = i_it_poly[ct_01]
        else:
            i_it_poly_compact = i_it_poly
        init_feature = get_gcn_feature(cnn_feature, i_it_poly_compact, ind, h, w)
        c_it_poly = c_it_poly * self.ro
        if self.with_img_idx:
            c_it_poly_reshape = c_it_poly.permute(0, 3, 2, 1) #(n_im, 2, n_vert, n_ct)
            init_feature_reshape = torch.zeros([c_it_poly.size(0),c_it_poly.size(1),init_feature.size(1),init_feature.size(2)]).to(c_it_poly.device) #(n_im, n_ct, n_feat, n_vert)
            init_feature_reshape[ct_01] = init_feature #(n_py, n_feat, n_vert)
            init_feature_reshape = init_feature_reshape.permute(0, 2, 3, 1) #(n_im, n_feat, n_vert, n_ct)
        else:
            init_feature_reshape = init_feature
            c_it_poly_reshape = c_it_poly.permute(0, 2, 1)

        init_input = torch.cat([init_feature_reshape, c_it_poly_reshape], dim=1)
        if return_vertex_classifier:
            offset, valid_logits = snake(init_input)
        else:
            offset = snake(init_input)
            valid_logits = None

        if self.with_img_idx:
            offset = offset.permute(0, 3, 2, 1) #(n_im, 2, n_vert, n_ct) > (n_im, n_ct, n_vert, 2)
        else:
            offset = offset.permute(0, 2, 1)
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

    def foward_train(self, output, batch, cnn_feature, num_points_init=128):
        ret = output
        init = self.prepare_training(output, batch, num_points=num_points_init)
        # if self.with_img_idx:
        #     # print(f"init['img_init_polys'] shape: {init['img_init_polys'].shape}")
        #     # print(f"init['ct_01'] shape : {init['ct_01'].shape}")
        #     # i_it_poly = torch.zeros([init['ct_01'].size(0), init['ct_01'].size(1), init['img_init_polys'].size(1), init['img_init_polys'].size(2)]).to(init['img_init_polys'].device)
        #     # i_it_poly[init['ct_01']] = init['img_init_polys']
        #     i_it_poly = init['img_init_polys']
        #     c_it_poly = torch.zeros_like(i_it_poly)
        #     c_it_poly[init['ct_01']] = init['can_init_polys']
        # else:
        i_it_poly = init['img_init_polys']
        c_it_poly = init['can_init_polys']
        py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, i_it_poly,
                                   c_it_poly, init['py_ind'], stride=self.evolve_stride, ct_01=init['ct_01'])
        py_preds = [py_pred]
        for i in range(self.iter):
            py_pred = py_pred / self.ro
            c_py_pred = img_poly_to_can_poly(py_pred)
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            extract_vtx_cls = False
            if self.use_vertex_classifier:
                if self.cfg.train.train_vtx_cls_all:
                    extract_vtx_cls = True
                elif i == (self.iter - 1):
                    extract_vtx_cls = True
            if extract_vtx_cls:
                py_pred, py_isvalid = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred,
                                           init['py_ind'], stride=self.evolve_stride, ct_01=init['ct_01'],
                                           return_vertex_classifier=True)
                ret.update({'py_valid_logits': py_isvalid})
            else:
                py_pred = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred,
                                           init['py_ind'], stride=self.evolve_stride, ct_01=init['ct_01'])
            py_preds.append(py_pred)
        ret.update({'py_pred': py_preds, 'img_gt_polys': init['img_gt_polys'] * self.ro,
                    'img_gt_init_polys': init['img_gt_init_polys']*self.ro,
                    'img_gt_coarse_polys': init['img_gt_coarse_polys']*self.ro})
        ret.update({'batch_ind': init['py_ind'].to(device=py_pred.device)})
        return output

    def foward_test(self, output, cnn_feature, ignore, extract_offset=False, num_points_init=128):
        ret = output
        with torch.no_grad():
            init = self.prepare_testing_init(output, num_points_init=num_points_init)
            img_init_polys = self.prepare_testing_evolve(init, cnn_feature.size(2), cnn_feature.size(3))
            if self.with_img_idx:
                ct_01 = torch.ones([cnn_feature.size(0), img_init_polys.size(0)], dtype=torch.bool)
                i_it_poly = torch.zeros([ct_01.size(0), ct_01.size(1), img_init_polys.size(1),
                                         img_init_polys.size(2)]).to(img_init_polys.device)
                i_it_poly[ct_01] = img_init_polys
                c_it_poly = torch.zeros_like(i_it_poly)
                c_it_poly[ct_01] = init['can_init_polys']
            else:
                i_it_poly = img_init_polys
                c_it_poly = init['can_init_polys']
                ct_01 = None
            if extract_offset:
                py, py_offset = self.evolve_poly(self.evolve_gcn, cnn_feature, i_it_poly, c_it_poly,
                                                 init['py_ind'],
                                                 ignore=ignore[0], stride=self.evolve_stride, extract_offset=True, ct_01=ct_01)
            else:
                py = self.evolve_poly(self.evolve_gcn, cnn_feature, i_it_poly, c_it_poly, init['py_ind'],
                                      ignore=ignore[0], stride=self.evolve_stride, ct_01=ct_01)
            if ct_01 is None:
                pys = [py, ]
            elif ct_01.numel() > 0:
                pys = [py[ct_01], ]
            else:
                pys = [py, ]
            if extract_offset:
                pys_offset = [py_offset, ]
            else:
                pys_offset = None
            for i in range(self.iter):
                py = py / self.ro
                c_py = img_poly_to_can_poly(py)
                evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                if extract_offset:
                    if (i == self.iter - 1) and self.use_vertex_classifier:
                        py, py_isvalid, py_offset = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['py_ind'],
                                              ignore=ignore[i + 1], stride=self.evolve_stride, extract_offset=True, ct_01=ct_01,
                                                                     return_vertex_classifier=True)
                        ret.update({'py_valid_logits': py_isvalid})
                    else:
                        py, py_offset = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['py_ind'],
                                                                     ignore=ignore[i + 1], stride=self.evolve_stride,
                                                                     extract_offset=True, ct_01=ct_01)
                    pys_offset.append(py_offset)
                else:
                    if (i == self.iter - 1) and self.use_vertex_classifier:
                        py, py_isvalid = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['py_ind'],
                                              ignore=ignore[i + 1], stride=self.evolve_stride, ct_01=ct_01,
                                                          return_vertex_classifier=True)
                        ret.update({'py_valid_logits': py_isvalid})
                    else:
                        py = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['py_ind'],
                                              ignore=ignore[i + 1], stride=self.evolve_stride, ct_01=ct_01)
                if ct_01 is None:
                    pys.append(py)
                elif ct_01.numel() > 0:
                    pys.append(py[ct_01])
                else:
                    pys.append(py)
            ret.update({'py': pys})
            ret.update({'batch_ind': init['py_ind'].to(device=py.device)})
            ret.update({'py_offset': pys_offset})
        return output

    def forward(self, output, cnn_feature, batch=None, test_stage='final', cfg=None):
        if batch is not None and 'test' not in batch['meta']:
            self.foward_train(output, batch, cnn_feature, num_points_init=cfg.commen.init_points_per_poly)
        else:
            ignore = [False] * (self.iter + 1)
            if test_stage == 'coarse' or test_stage == 'init':
                ignore = [True for _ in ignore]
            if test_stage == 'final':
                ignore[-1] = True
            self.foward_test(output, cnn_feature, ignore=ignore, extract_offset=cfg.test.extract_offset if cfg is not None else False, num_points_init=cfg.commen.init_points_per_poly)
        return output

