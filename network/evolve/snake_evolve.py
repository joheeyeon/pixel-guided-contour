import torch.nn as nn
from .snake import Snake
from .utils import prepare_training_snake, prepare_testing_snake_init, img_poly_to_can_poly, get_gcn_feature, prepare_training_snake_evolve, prepare_testing_snake_evolve, get_adj_ind
import torch


class Evolution(nn.Module):
    def __init__(self, evole_ietr_num=3, evolve_stride=1., ro=4., with_img_idx=False, refine_kernel_size=1, use_part='all', cfg=None):
        super(Evolution, self).__init__()
        assert evole_ietr_num >= 1
        self.evolve_stride = evolve_stride
        self.ro = ro
        self.use_part = use_part
        self.cfg = cfg

        if self.use_part in ('init','all','coarse'):
            self.fuse = nn.Conv1d(128, 64, 1)
            self.init_gcn = Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid', with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size)
        if self.use_part in ('all','coarse'):
            self.evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid', with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size)
            self.iter = evole_ietr_num - 1
            self.with_img_idx = with_img_idx
        if self.use_part in ('all',):
            for i in range(self.iter):
                evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid', with_img_idx=with_img_idx, refine_kernel_size=refine_kernel_size)
                self.__setattr__('evolve_gcn'+str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = prepare_training_snake(output, batch)
        output.update({'img_it_init_polys': init['img_it_init_polys'], 'img_it_polys': init['img_it_polys']})
        output.update({'img_gt_init_polys': init['img_gt_init_polys'], 'img_gt_polys': init['img_gt_polys']})
        return init

    def prepare_training_evolve(self, output, batch, init):
        evolve = prepare_training_snake_evolve(output['ex_pred'], init, train_nearest_gt=self.cfg.train.evolve_params['train_nearest_gt'] if 'train_nearest_gt' in self.cfg.train.evolve_params else True)
        output.update({'img_it_polys': evolve['img_it_polys'], 'can_it_polys': evolve['can_it_polys'], 'img_gt_polys': evolve['img_gt_polys']})
        evolve.update({'py_ind': init['py_ind']})
        return evolve

    def prepare_testing_init(self, output):
        init = prepare_testing_snake_init(output['detection'][..., :4], output['detection'][..., 4], th_ct_score=self.cfg.test.ct_score) #[TO-DO] py_ind=output['py_ind']
        output['detection'] = output['detection'][output['detection'][..., 4] > self.cfg.test.ct_score]
        output.update({'it_ex': init['img_it_init_polys']})
        return init

    def prepare_testing_evolve(self, output, h, w):
        ex = output['ex']
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w-1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h-1)
        evolve = prepare_testing_snake_evolve(ex)
        output.update({'img_it_polys': evolve['img_it_polys']})
        return evolve

    def init_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, adj_num=4, init_poly_num=40):
        if len(i_it_poly) == 0:
            return torch.zeros([0, 4, 2]).to(i_it_poly)

        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        center = (torch.min(i_it_poly, dim=1)[0] + torch.max(i_it_poly, dim=1)[0]) * 0.5
        ct_feature = get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
        init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature)], dim=1)
        init_feature = self.fuse(init_feature)

        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1) # init_feature : 64
        adj = get_adj_ind(adj_num, init_input.size(2), init_input.device)
        i_poly = i_it_poly + snake(init_input).permute(0, 2, 1)
        i_poly = i_poly[:, ::init_poly_num//4]

        return i_poly

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, adj_num=4):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        c_it_poly = c_it_poly * self.ro
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = get_adj_ind(adj_num, init_input.size(2), init_input.device)
        i_poly = i_it_poly * self.ro + snake(init_input).permute(0, 2, 1)
        return i_poly

    def forward(self, output, cnn_feature, batch=None):
        if batch is not None and 'test' not in batch['meta']:
            if self.use_part in ('init', 'all', 'coarse'):
                with torch.no_grad():
                    init = self.prepare_training(output, batch)  # get "init" info of gt

                ex_pred = self.init_poly(self.init_gcn, cnn_feature, init['img_it_init_polys'], init['can_it_init_polys'], init['4py_ind'])
                output.update({'ex_pred': ex_pred, 'img_gt_init_polys': output['img_gt_init_polys']})
            if self.use_part in ('all', 'coarse'):
                with torch.no_grad():
                    init = self.prepare_training_evolve(output, batch, init)
                py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, init['img_it_polys'], init['can_it_polys'],
                                           init['py_ind'])  # using start point from detected box or gt box
                py_preds = [py_pred]
            if self.use_part == 'all':
                for i in range(self.iter):
                    py_pred = py_pred / self.ro
                    c_py_pred = img_poly_to_can_poly(py_pred)
                    evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                    py_pred = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'])
                    py_preds.append(py_pred)

            if self.use_part in ('all', 'coarse'):
                output.update({'py_pred': py_preds, 'img_gt_polys': output['img_gt_polys'] * self.cfg.commen.down_ratio})
            output.update({'batch_ind': init['py_ind'].to(py_preds[-1].device)})
        else:
            with torch.no_grad():
                if self.use_part in ('init', 'all', 'coarse'):
                    init = self.prepare_testing_init(output)
                    ex = self.init_poly(self.init_gcn, cnn_feature, init['img_it_init_polys'], init['can_it_init_polys'], init['ind'])
                    output.update({'ex': ex})
                if self.use_part in ('all', 'coarse'):
                    evolve = self.prepare_testing_evolve(output, cnn_feature.size(2), cnn_feature.size(3))
                    py = self.evolve_poly(self.evolve_gcn, cnn_feature, evolve['img_it_polys'], evolve['can_it_polys'],
                                          init['ind'])
                    pys = [py / self.ro]
                    offsets = [pys[0] - evolve['img_it_polys']]
                if self.use_part == 'all':
                    for i in range(self.iter):
                        pre_py = py / self.ro
                        c_py = img_poly_to_can_poly(pre_py)
                        evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                        py = self.evolve_poly(evolve_gcn, cnn_feature, pre_py, c_py, init['ind'])
                        pys.append(py / self.ro)
                        offsets.append(pys[i + 1] - pys[i])

                if self.use_part in ('all', 'coarse'):
                    output.update({'py': pys, 'py_offset': offsets})
                output.update({'batch_ind': init['ind'].to(pys[-1].device)})
        return output

