from configs.base import commen, data, model, train, test
import numpy as np

commen.task = 'ccp'

data.valid_box_margin = 0
data.valid_box_area = 0
data.scale_range = [1.0, 1.0]
data.scale = None
data.augment_shift = False
data.test_scale = (416, 416)
data.input_w, data.input_h = (416, 416)
data.add_augment_curved = False
data.douglas.update({'D': 0.01})
data.gt_mask_label = {255: 1}

model.ccp_deform_pixel_norm = 'softmax'
model.use_refine_pixel = False
model.refine_pixel_param.update({'module_structure': {'conv_1': [256, 256]}})
model.ccp_refine_pixel_as_residual =False
model.contour_map_down_sample = False
model.ccp_with_proj = False
model.ccp_refine_with_pre_pixel = False
model.heads['ct_hm'] = 1
model.heads.update({'pixel': 2})
model.evolve_use_input_with_rel_ind = True
model.evolve_iters = 3
model.gcn_weight_sharing = False

train.save_ep = 1
train.optimizer = {'name': 'adam', 'lr': 1e-4,
                 'weight_decay': 1e-4, 'scheduler': 'MultiStepLR',
                 'milestones': [500, ],
                 'gamma': 0.5}
train.val_metric = 'ap'
train.best_metric_crit = 'max'
train.epoch = 1000
train.earlystop = 20
train.min_epochs_for_earlystop = 50
train.dataset = 'pcb_train_622_v2_remove_same_add_curv4_0.2angle4'
train.weight_dict.update({'cv_evolve_2': 1., 'tv_evolve_0': 0.01, 'tv_evolve_1': 1., 'tv_evolve_2': 0.1, 'tv_evolve': 0., 'tv_coarse': 0., 'tv_init': 0., 'evolve': 3., 'pixel': 40.})
train.loss_type.update({'tv': 'l1', 'pixel': 'focal', 'cv': 'l1'})
train.loss_params['pixel'].update({'gamma': 2})
train.is_normalize_pixel = False

train.ml_range_py = 'except_1st'
train.with_dml = True
train.ml_start_epoch = 5

train.use_dp = False

test.dataset = 'pcb_val_622_v2_remove_same'
test.with_nms = False

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test