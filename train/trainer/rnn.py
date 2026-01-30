import torch.nn as nn
from .utils import FocalLoss, sigmoid, DMLoss, TVLoss, CurvLoss, WeightedPYLoss, MDLoss, EdgeStandardDeviationLoss
import torch
import torch.nn.functional as F

LOSS_DICT = {'l1' : 'l1',
             'smooth_l1': 'smooth_l1',
             'l2': 'mse'}

class NetworkWrapper(nn.Module):
    def __init__(self, net, with_dml=True, ml_start_epoch=10, weight_dict=None, cfg=None):
        super(NetworkWrapper, self).__init__()
        self.iter = 0
        self.cfg = cfg
        self.net = net
        self.with_dml = with_dml
        self.ml_start_epoch = ml_start_epoch
        self.weight_dict = weight_dict

        if 'pixel' in self.cfg.model.heads:
            if self.cfg.model.heads['pixel'] == 1:
                self.pix_crit = FocalLoss()
            else:
                self.pix_crit = torch.nn.functional.cross_entropy
        self.ct_crit = FocalLoss()
        flag_def = True
        if cfg is not None:
            if 'weighted' in cfg.train.loss_type['py']:
                self.py_crit = WeightedPYLoss(type=cfg.train.loss_type['py'], th_weight=cfg.train.loss_params['py']['th_weight'],
                                              under_weight=cfg.train.loss_params['py']['under_weight'] if 'under_weight' in cfg.train.loss_params['py'] else 0.)
            else:
                self.py_crit = getattr(torch.nn.functional,f"{LOSS_DICT[cfg.train.loss_type['py']]}_loss")
        else:
            self.py_crit = getattr(torch.nn.functional, f"smooth_l1_loss")

        # self.py_crit_0 = self.py_crit if cfg.train.loss_type['py'] == 'smooth_l1' else torch.nn.functional.smooth_l1_loss #until240408.11:52
        self.py_crit_0 = self.py_crit if 'smooth_l1' in cfg.train.loss_type['py'] else torch.nn.functional.smooth_l1_loss #since240408.11:52
        self.tv_crit = TVLoss(type=cfg.train.loss_type['tv'] if cfg is not None else 'smooth_l1')
        self.cv_crit = CurvLoss(type=cfg.train.loss_type['cv'] if cfg is not None else 'smooth_l1')

        if with_dml:
            self.ml_crit = DMLoss(type=cfg.train.loss_type['dm'] if cfg is not None else 'smooth_l1')
        elif cfg.train.with_mdl:
            self.ml_crit = MDLoss(type=cfg.train.loss_type['md'] if cfg is not None else 'smooth_l1', match_with_ini=cfg.train.ml_match_with_ini if cfg is not None else True)
        else:
            self.ml_crit = self.py_crit

        PY_RANGE_DICT = {'final': [self.cfg.model.evolve_iters - 1],
                         'last2': [i for i in range(self.cfg.model.evolve_iters - 2, self.cfg.model.evolve_iters)],
                         'all': [i for i in range(self.cfg.model.evolve_iters)]}
        self.ml_range_py = PY_RANGE_DICT[self.cfg.train.ml_range_py]
        if ('edge_std' in self.cfg.train.weight_dict) or ('edge_std_init' in self.cfg.train.weight_dict) or ('edge_std_coarse' in self.cfg.train.weight_dict) or ('edge_std_evolve' in self.cfg.train.weight_dict):
            self.eeq_crit = EdgeStandardDeviationLoss()
        else:
            self.eeq_crit = None

    def forward(self, batch, mode='default'):
        if 'test' in batch['meta']:
            output = self.net(batch['inp'], batch=batch)
            return output
        else:
            out_ontraining = {}
            output = self.net(batch['inp'], batch=batch)
            epoch = batch['epoch']
            scalar_stats = {}
            loss = 0.

            keyPointsMask = batch['keypoints_mask'][batch['ct_01']]
            # pixel
            if 'pixel' in self.cfg.model.heads:
                pixel_gt = F.interpolate(batch['pixel_gt'].unsqueeze(1).float(),
                                         size=(output['pixel'].size(-2), output['pixel'].size(-1)), mode='nearest').squeeze(1)
                if self.cfg.model.heads['pixel'] == 1:
                    pix_loss = self.pix_crit(sigmoid(output['pixel']), pixel_gt.bool().float())
                else:
                    pix_loss = self.pix_crit(output['pixel'], pixel_gt.bool().long())
                scalar_stats.update({'pix_loss': pix_loss})
                weight_pix = 1. if 'pixel' not in self.weight_dict else self.weight_dict['pixel']
                loss += weight_pix * pix_loss
            # ct
            if self.weight_dict['box_ct'] > 0:
                ct_loss = self.ct_crit(sigmoid(output['ct_hm']), batch['ct_hm'])
                scalar_stats.update({'ct_loss': ct_loss})
                loss += self.weight_dict['box_ct'] * ct_loss
            # init & coarse
            num_polys = len(output['poly_init'])
            if num_polys == 0:
                init_py_loss = torch.sum(output['poly_init']) * 0.
                coarse_py_loss = torch.sum(output['poly_coarse']) * 0.
            else:
                init_py_loss = getattr(self, 'py_crit' if epoch >= self.cfg.train.epoch_py_crit_0 else 'py_crit_0')(output['poly_init'], output['img_gt_init_polys'])
                coarse_py_loss = getattr(self, 'py_crit' if epoch >= self.cfg.train.epoch_py_crit_0 else 'py_crit_0')(output['poly_coarse'], output['img_gt_coarse_polys'])
            if self.weight_dict['init'] > 0:
                scalar_stats.update({'init_py_loss': init_py_loss})
                loss += init_py_loss * self.weight_dict['init']
            if self.weight_dict['coarse'] > 0:
                scalar_stats.update({'coarse_py_loss': coarse_py_loss})
                loss += coarse_py_loss * self.weight_dict['coarse']

            # for deform (rnn)
            if self.weight_dict['evolve'] > 0:
                py_loss = 0
                n = len(output['py_pred']) - 1 if self.with_dml or self.cfg.train.with_mdl else len(output['py_pred'])
                tmp_loss = {'inp': batch['inp'], 'img_gt_polys': output['img_gt_polys'], 'batch_ind': output['batch_ind']}
                for i in range(n):
                    if num_polys == 0:
                        part_py_loss = torch.sum(output['py_pred'][i]) * 0.0
                        py_loss += part_py_loss / len(output['py_pred'])
                        scalar_stats.update({'py_loss{}'.format(i): part_py_loss})
                    elif self.with_dml or self.cfg.train.with_mdl:
                        if i not in self.ml_range_py:
                            # part_py_loss = getattr(self,
                            #                        'py_crit' if epoch >= self.cfg.train.epoch_py_crit_0 else 'py_crit_0')(
                            #     output['py_pred'][i], output['img_gt_polys'])
                            part_py_loss = getattr(self, 'py_crit' if epoch >= self.cfg.train.epoch_py_crit_0 else 'py_crit_0')(output['py_pred'][i], output['img_gt_polys'], reduction='none') #tmp
                            tmp_loss.update({f'py_loss{i}': part_py_loss, f'py_pred{i}': output['py_pred'][i]})
                            part_py_loss = part_py_loss.mean()
                            py_loss += part_py_loss / len(output['py_pred'])
                            scalar_stats.update({'py_loss{}'.format(i): part_py_loss})
                    else:
                        part_py_loss = getattr(self, 'py_crit' if epoch >= self.cfg.train.epoch_py_crit_0 else 'py_crit_0')(
                            output['py_pred'][i], output['img_gt_polys'], reduction='none')  # tmp
                        tmp_loss.update({f'py_loss{i}': part_py_loss, f'py_pred{i}': output['py_pred'][i]})
                        part_py_loss = part_py_loss.mean()
                        py_loss += part_py_loss / len(output['py_pred'])
                        scalar_stats.update({'py_loss{}'.format(i): part_py_loss})

                loss += py_loss * self.weight_dict['evolve']

                for i in self.ml_range_py:
                    if self.with_dml:
                        if epoch >= self.ml_start_epoch:
                            # dm_loss = self.dml_crit(output['py_pred'][-2],
                            #                         output['py_pred'][-1],
                            #                         output['img_gt_polys'],
                            #                         keyPointsMask)
                            dm_loss = self.ml_crit(output['py_pred'][i-1],
                                                    output['py_pred'][i],
                                                    output['img_gt_polys'],
                                                    keyPointsMask, all_loss=tmp_loss)
                            dm_loss = dm_loss.mean()
                            scalar_stats.update({f'py_loss{i}': dm_loss})
                            loss += dm_loss / len(output['py_pred']) * self.weight_dict['evolve']
                        else:
                            # py_last_loss = getattr(self, 'py_crit' if epoch >= self.cfg.train.epoch_py_crit_0 else 'py_crit_0')(output['py_pred'][-1], output['img_gt_polys'])
                            py_last_loss = getattr(self,
                                                   'py_crit' if epoch >= self.cfg.train.epoch_py_crit_0 else 'py_crit_0')(
                                output['py_pred'][i], output['img_gt_polys'], reduction='none')
                            tmp_loss.update({'py_final_loss': py_last_loss, 'py_pred_final': output['py_pred'][-1]})
                            py_last_loss = py_last_loss.mean()
                            scalar_stats.update({f'py_loss{i}': py_last_loss})
                            loss += py_last_loss / len(output['py_pred']) * self.weight_dict['evolve']
                    elif self.cfg.train.with_mdl:
                        if epoch >= self.ml_start_epoch:
                            # dm_loss = self.dml_crit(output['py_pred'][-2],
                            #                         output['py_pred'][-1],
                            #                         output['img_gt_polys'],
                            #                         keyPointsMask)
                            md_loss = self.ml_crit(output['py_pred'][i-1],
                                                    output['py_pred'][i],
                                                    output['img_gt_polys'], all_loss=tmp_loss)
                            md_loss = md_loss.mean()
                            scalar_stats.update({f'py_loss{i}': md_loss})
                            loss += md_loss / len(output['py_pred']) * self.weight_dict['evolve']
                        else:
                            # py_last_loss = getattr(self, 'py_crit' if epoch >= self.cfg.train.epoch_py_crit_0 else 'py_crit_0')(output['py_pred'][-1], output['img_gt_polys'])
                            py_last_loss = getattr(self,
                                                   'py_crit' if epoch >= self.cfg.train.epoch_py_crit_0 else 'py_crit_0')(
                                output['py_pred'][i], output['img_gt_polys'], reduction='none')
                            tmp_loss.update({'py_final_loss': py_last_loss, 'py_pred_final': output['py_pred'][-1]})
                            py_last_loss = py_last_loss.mean()
                            scalar_stats.update({f'py_loss{i}': py_last_loss})
                            loss += py_last_loss / len(output['py_pred']) * self.weight_dict['evolve']

            ## total variation
            if ('tv' in self.weight_dict) or ('tv_coarse' in self.weight_dict) or ('tv_init' in self.weight_dict) or ('tv_evolve' in self.weight_dict):
                if 'tv_init' in self.weight_dict:
                    weight_tv_init = self.weight_dict['tv_init']
                else:
                    weight_tv_init = self.weight_dict['init'] * self.weight_dict['tv']
                if 'tv_coarse' in self.weight_dict:
                    weight_tv_coarse = self.weight_dict['tv_coarse']
                else:
                    weight_tv_coarse = self.weight_dict['coarse'] * self.weight_dict['tv']
                if 'tv_evolve' in self.weight_dict:
                    weight_tv_evolve = self.weight_dict['tv_evolve']
                else:
                    weight_tv_evolve = self.weight_dict['evolve'] * self.weight_dict['tv']

                if num_polys == 0:
                    init_tv_loss = torch.sum(output['poly_init']) * 0.
                    coarse_tv_loss = torch.sum(output['poly_coarse']) * 0.
                    evolve_tv_loss = torch.sum(output['py_pred']) * 0.
                else:
                    init_tv_loss = self.tv_crit(output['poly_init'])
                    coarse_tv_loss = self.tv_crit(output['poly_coarse'])
                    evolve_tv_loss = 0.
                    for i in range(len(output['py_pred'])):
                        evolve_tv_loss += self.tv_crit(output['py_pred'][i])/len(output['py_pred'])

                if weight_tv_init > 0:
                    scalar_stats.update({'init_tv_loss': init_tv_loss})
                    loss += init_tv_loss * weight_tv_init
                if weight_tv_coarse > 0:
                    scalar_stats.update({'coarse_tv_loss': coarse_tv_loss})
                    loss += coarse_tv_loss * weight_tv_coarse
                if weight_tv_evolve > 0:
                    scalar_stats.update({'evolve_tv_loss': evolve_tv_loss})
                    loss += evolve_tv_loss * weight_tv_evolve

            if ('cv' in self.weight_dict) or ('cv_coarse' in self.weight_dict):
                if self.weight_dict['cv'] > 0:
                    if num_polys == 0:
                        init_cv_loss = torch.sum(output['poly_init']) * 0.
                        coarse_cv_loss = torch.sum(output['poly_coarse']) * 0.
                        evolve_cv_loss = torch.sum(output['py_pred']) * 0.
                    else:
                        init_cv_loss = self.cv_crit(output['poly_init'])
                        coarse_cv_loss = self.cv_crit(output['poly_coarse'])
                        evolve_cv_loss = 0.
                        for i in range(len(output['py_pred'])):
                            evolve_cv_loss += self.cv_crit(output['py_pred'][i]) / len(output['py_pred'])
                    scalar_stats.update({'init_cv_loss': init_cv_loss})
                    scalar_stats.update({'coarse_cv_loss': coarse_cv_loss})
                    scalar_stats.update({'evolve_cv_loss': evolve_cv_loss})
                    if 'cv_init' in self.weight_dict:
                        weight_cv_init = self.weight_dict['cv_init']
                    else:
                        weight_cv_init = self.weight_dict['init'] * self.weight_dict['cv']
                    if 'cv_coarse' in self.weight_dict:
                        weight_cv_coarse = self.weight_dict['cv_coarse']
                    else:
                        weight_cv_coarse = self.weight_dict['coarse'] * self.weight_dict['cv']
                    if 'cv_evolve' in self.weight_dict:
                        weight_cv_evolve = self.weight_dict['cv_evolve']
                    else:
                        weight_cv_evolve = self.weight_dict['evolve'] * self.weight_dict['cv']
                    loss += init_cv_loss * weight_cv_init
                    loss += coarse_cv_loss * weight_cv_coarse
                    loss += evolve_cv_loss * weight_cv_evolve

            ## edge standard deviation loss (Edge Equal loss = eeq loss)
            if self.eeq_crit is not None:
                if num_polys == 0:
                    init_eeq_loss = torch.sum(output['poly_init']) * 0.
                    coarse_eeq_loss = torch.sum(output['poly_coarse']) * 0.
                    evolve_eeq_loss = torch.sum(output['py_pred']) * 0.
                else:
                    init_eeq_loss = self.eeq_crit(output['poly_init'])
                    coarse_eeq_loss = self.eeq_crit(output['poly_coarse'])
                    evolve_eeq_loss = 0.
                    for i in range(len(output['py_pred'])):
                        evolve_eeq_loss += self.eeq_crit(output['py_pred'][i]) / len(output['py_pred'])

                if 'edge_std_init' in self.weight_dict:
                    weight_eeq_init = self.weight_dict['edge_std_init']
                else:
                    weight_eeq_init = self.weight_dict['init'] * self.weight_dict['edge_std']
                if 'edge_std_coarse' in self.weight_dict:
                    weight_eeq_coarse = self.weight_dict['edge_std_coarse']
                else:
                    weight_eeq_coarse = self.weight_dict['coarse'] * self.weight_dict['edge_std']
                if 'edge_std_evolve' in self.weight_dict:
                    weight_eeq_evolve = self.weight_dict['edge_std_evolve']
                else:
                    weight_eeq_evolve = self.weight_dict['evolve'] * self.weight_dict['edge_std']

                if weight_eeq_init > 0:
                    scalar_stats.update({'init_eeq_loss': init_eeq_loss})
                    loss += init_eeq_loss * weight_eeq_init
                if weight_eeq_coarse > 0:
                    scalar_stats.update({'coarse_eeq_loss': coarse_eeq_loss})
                    loss += coarse_eeq_loss * weight_eeq_coarse
                if weight_eeq_evolve > 0:
                    scalar_stats.update({'evolve_eeq_loss': evolve_eeq_loss})
                    loss += evolve_eeq_loss * weight_eeq_evolve

            scalar_stats.update({'loss': loss})

            #tmp
            # import os
            # if not os.path.exists(f'{self.cfg.commen.record_dir}/tmp_loss'):
            #     os.makedirs(f'{self.cfg.commen.record_dir}/tmp_loss', exist_ok=True)
            # save_as_npy(tmp_loss, path=f'{self.cfg.commen.record_dir}/tmp_loss/{self.iter}.mat')

            self.iter += 1
            return output, loss, scalar_stats, out_ontraining

def save_as_npy(dict_to_save, path='dict.mat'):
    from scipy.io import savemat
    for it_torch in dict_to_save:
        dict_to_save[it_torch] = dict_to_save[it_torch].clone().detach().cpu().numpy()

    savemat(path, dict_to_save)