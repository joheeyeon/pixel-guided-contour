from configs.base import commen, data, model, train, test
import numpy as np

commen.task = 'ccp'

# commen.init_points_per_poly = 64
# commen.points_per_poly = 64
# data.points_per_poly = 64
# model.points_per_poly = 64
# model.heads['wh'] = commen.init_points_per_poly * 2

data.valid_box_margin = 0
data.valid_box_area = 0
scale = np.array([1024, 1024])
data.augment_shift = True
data.test_scale = (512, 512)
data.input_w, data.input_h = (512, 512)
data.add_augment_curved = False
data.douglas.update({'D': 0.1})
data.gt_mask_label = {255: 1}

model.ccp_deform_pixel_norm = 'softmax'
model.use_refine_pixel = False
model.ccp_with_proj = False
model.ccp_refine_with_pre_pixel = True
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
train.earlystop = 10
train.min_epochs_for_earlystop = 50
train.dataset = 'whu_train'
train.weight_dict.update({'cv_evolve_2': 0.00001, 'tv_evolve_0': 0.01, 'tv_evolve_1': 1., 'tv_evolve_2': 1., 'tv_evolve': 0., 'tv_coarse': 0., 'tv_init': 0., 'evolve': 3., 'pixel': 0.5})
train.loss_type.update({'cv': 'l1', 'tv': 'l1', 'pixel': 'focal'})
train.loss_params['pixel'].update({'gamma': 2})
train.is_normalize_pixel = False

train.with_dml = True
train.ml_start_epoch = 5

train.use_dp = False

test.dataset = 'whu_val'
test.with_nms = False

commen.down_ratio = 1
train.down_ratio = 1
model.down_ratio = 4
data.down_ratio = 1
test.down_ratio = 1
model.concat_upper_layer = 'base_0'

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test