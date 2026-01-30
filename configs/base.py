import numpy as np

class commen(object):
    task = 'e2ec' #init
    points_per_poly_steps = None
    init_points_per_poly = 128
    points_per_poly = 128
    down_ratio = 4
    result_dir = 'data/result'
    record_dir = 'data/record'
    model_dir = 'data/model'
    output_dir = 'None'
    seed = 42
    deterministic_mode = "never"

class data(object):
    augment_rotate = None #'random','all_4'
    gt_mask_label = {}
    augment_shift = True
    valid_box_margin = 10
    valid_box_area = 1000
    douglas = {}
    get_keypoints_mask = True
    class_type = 'Dataset'
    contour_param = {}
    add_augment_curved = False
    rdp_th_n_pts = 18
    rdp_eps = 4.
    flip_type = 'lr'
    val_split = 'val'
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    data_rng = np.random.RandomState(123)
    eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                       dtype=np.float32)
    eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    down_ratio = commen.down_ratio
    scale = np.array([512, 512])
    input_w, input_h = (512, 512)
    test_scale = None
    scale_range = [0.6, 1.4]
    points_per_poly = commen.points_per_poly

class model(object):
    refine_use_trans_feature = True
    vtx_cls_kernel_size = 1
    vtx_cls_common_prediction_type = 1
    contour_map_down_ratio = 2
    ccp_refine_pixel_as_residual = False
    contour_map_down_sample = False
    ccp_deform_pixel_norm = 'unnormalized' #('softmax','argmax')
    ccp_refine_pixel_input_norm = 'unnormalized' #('softmax','argmax')
    ccp_refine_with_pre_pixel = False
    ccp_with_proj = True
    ccp_dim_out_proj = 64
    use_vertex_classifier = False
    use_bn_in_head = True
    norm_num_groups = 32
    norm_type = 'batch'
    concat_upper_layer = None
    refine_pixel_param = {}
    use_refine_pixel = True
    cat_feature_normalized = False
    cat_include_coarse = False # 이전에는 True 였음 (rev 25-09-04)
    cut_grad_add_feature = False
    cat_feature_with_pixelmap = False
    dla_clone_to_ida_up = True
    teacher_cfg_name = None
    path_pretrained_teacher = None
    dla_pretrained = True
    type_add_pixel_mask = 'none'
    with_sharp_contour = False
    sharp_param = {'dim_reduce_after_cat': False,
                   'ipc_num_params': [8, 8, 1],
                   'fine_dim': 16,
                   'ipc_random_sample_range': 10.,
                   'ipc_match_dist_p': 2.,
                   'ipc_dynamic': True,
                   'refine_iters': 3,
                   'refine_constant': 0.001,
                   'py_refine_init': 'add_zero',
                   'refine_normal_outward': False}
    raster_scale = 1.
    with_rasterize_net = False
    is_raster_down_sampled = True
    raster_netparams = {
        'dec_kernel': 3,
        'skipp_connection': False,
        'reg': True,
        'latent_resolution': 13,
        'dec_up_conv_kernel': 2,
        'dec_up_conv_stride': 2,
        'enc_channel_list': [32, 64, 128, 169],
        'dec_channel_list': [64, 64, 32, 32, 16, 16],
        'dec_up_conv_inds': [0, 2, 4],
        'use_bn': False,
        'use_final_relu': True
    }
    raster_sigma = 1.
    refine_kernel_size = 3
    with_img_idx = False
    raster_type = 'xy_as_channel'
    grad_feature_params = {'sigma' : 2}
    add_grad_feature = False
    evolve_grad_feature_neighbors = True
    rnn_params = {}
    lstm_fc_activation_type = 'none'
    lstm_fc_type = 'dim_reduction_first'
    lstm_input_window_stride = 1
    lstm_input_window_size = (7, 7)
    lstm_type = 'linear'
    lstm_fc_dims = [2]
    lstm_bidirectional = False
    evolve_is_dec_exclusive = True
    evolve_use_input_with_rel_ind = False
    lstm_hidden_type = 'dynamic_hidden_dim'
    lstm_n_layer = 2
    lstm_hidden_dims = [128,32]
    load_vgg = True
    dla_layer = 34
    head_conv = 256
    # head_conv 구조 세부 설정 (kernel size, layer 수, filter 갯수 모두 설정 가능)
    head_conv_config = {
        'kernel_sizes': [3, 1],     # 각 layer의 kernel size 리스트 [intermediate, final]
        'channels': None,           # None이면 기본 [head_conv, classes] 사용
        'use_relu': [True],         # 각 layer 후 ReLU 사용 여부 (마지막 layer는 자동으로 False)
        'padding': 'auto'           # 'auto'면 (kernel_size-1)//2로 자동 계산
    }
    use_dcn = True
    points_per_poly = commen.points_per_poly
    down_ratio = commen.down_ratio
    init_stride = 10.
    coarse_stride = 4.
    evolve_stride = 1.
    evolve_iter_num = 1  # GCN iteration 횟수
    gcn_weight_sharing = True  # True: weight sharing, False: non-sharing
    use_pixel_on_init = False  # True: pixel head 먼저 수행 후 concat, False: 표준 동시 생성
    pixel_concat_with_activation = False  # True: pixel output에 softmax 적용 후 concat, False: raw output concat
    use_3x3_feature = False  # True: use 3x3 patch features around vertices, False: use 1x1 features
    feature_3x3_mode = 'flatten'  # 'flatten': flatten 3x3 to C*9 channels, 'conv2d': use 2D circular convolution
    feature_3x3_detach = True  # True: detach 3x3 features (no gradient) - safer for AsStridedBackward0 issues, False: keep gradient
    concat_multi_layers = None  # 새로운 multi-scale concat 옵션: ['base_2', 'base_3'] 등 - 여러 레이어 동시 concat
    backbone_num_layers = 34
    heads = {'ct_hm': 20, 'wh': commen.init_points_per_poly * 2}
    # heads = {'ct_hm': 20, 'wh': commen.points_per_poly * 2}
    evolve_iters = 3

class train(object):
    stage = 2
    train_vtx_cls_all = False
    apply_amp = False
    validate_with_reduction = False
    use_dp = True
    is_normalize_pixel = True
    kd_param = {'losses': {},
                'reg_margin': 0,
                'reg_condition_type': 'error',
                'soft_T': 10.,
                'weight_type': 'un-normalized'}
    sharp_param = {'n_iter_train_ipc_rancom_sample': 5,
                   'train_gt': True,
                   'fix_step_number': True,
                   'step_num_init': 1,
                   'ipc_start_epoch': 0,
                   'avg_ipc_random_loss': False,
                   'train_with_refine': False,
                   'refine_start_epoch': 0,
                   'refine_with_dml': False}
    evolve_params = {}
    save_ontraining = {}
    with_iou_loss = False
    iou_params = {'use_keypoints': False,
                  # 'num_keypoints': 32,} #rev 24-09-08
                  'filter_type': None,
                  'schedule_type': None}
    validate_first = True
    start_epoch_region = 0
    load_trained_rasterizer = 'data/model/rasterizer/rasterizer.pth'
    raster_add_range_init_vertex = [5, 20]
    contour_batch_size = 24
    raster_add_random_contour = False
    dml_range = 'none'
    mdml_range = 'none'
    ml_range_py = 'final'
    ml_match_with_ini = True
    loss_params = {'py':{}, 'pixel':{}, 'vertex_cls':{'train_only_final': True}}
    fix_deform = False
    fix_decode = False
    fix_dla = False
    load_trained_model = None
    epoch_py_crit_0 = 5
    loss_type = {'dm': 'smooth_l1', 'md': 'smooth_l1', 'tv': 'smooth_l1', 'cv': 'l2', 'py': 'smooth_l1', 'raster': None}
    with_mdl = False
    with_dml = False
    use_gt_det = False
    val_metric = 'ap'
    best_metric_crit = 'max'
    earlystop = 150
    min_epochs_for_earlystop = 0  # early stopping이 활성화되기 전 최소 에포크 수
    is_save_model_all = False
    save_ep = 5
    # eval_ep = 5
    optimizer = {'name': 'adam', 'lr': 1e-4,
                 'weight_decay': 5e-4,
                 'scheduler': 'MultiStepLR',
                 'milestones': [80, 120, ],
                 'gamma': 0.5}
    
    # Temperature parameter scheduling 옵션 (trainable_softmax, trainable_softmax_softclamp 사용 시)
    temperature_advanced_scheduling = {
        'enabled': False,                    # 고급 스케줄링 사용 여부 (기본: False)
        'freeze_epochs': 1,                  # 초반 freeze할 epoch 수 (기본: 1)
        'warmup_ratio': 0.03,                # warmup 비율 (전체 epoch 대비, 기본: 3%)
        'final_scheduler': 'cosine',         # warmup 후 사용할 스케줄러 ('cosine' or 'multistep')
        'cosine_eta_min_ratio': 1e-3,       # cosine의 경우 eta_min = initial_lr * cosine_eta_min_ratio
    }
    
    batch_size = 24
    num_workers = 8
    epoch = 150
    ml_start_epoch = 10
    dml_start_epoch = 0
    mdml_start_epoch = 0
    weight_dict = {'box_ct': 1, 'init': 0.1, 'coarse': 0.1, 'evolve': 1., 'tv': 0., 'cv': 0.}
    dataset = 'sbd_train'
    # ccp 및 ccp_maskinit stage 1에서 ct_hm head도 함께 학습할지 여부 (기본값: False - pixel head만 학습)
    stage1_train_ct_hm = False
    # ccp stage 1에서 wh head도 함께 학습할지 여부 (기본값: False - pixel head만 학습)
    stage1_train_wh = False
    # ccp 및 ccp_maskinit stage 2에서 stage 1 모듈을 freeze할지 여부 (기본값: False - 모든 모듈 학습)
    freeze_s1_modules = False
    # ccp_maskinit GT-Prediction 매칭 시 최대 center distance (피처맵 좌표계 기준, 기본값: 50픽셀)
    max_center_distance_maskinit = 50.0
    # Stage 2 학습 시 Stage 1 모델 selective loading 사용 여부 (기본값: True)
    # True: 전체 모델 로드 실패 시 Stage 1 모듈만 선택적 로드, False: 전체 모델 로드만 시도
    use_selective_s1_loading = True

class test(object):
    with_vertex_reordering = False
    check_val_every_n_epoch = 1
    track_self_intersection = False  # 기본 꺼두고 필요할 때 켬 (edit:feat:self-intersection-count:25-08-10)
    track_include_reduced = True  # py_reduced까지 포함할지 (edit:feat:self-intersection-count:25-08-10)
    viz_mode = "final"
    visualize = False
    # ccp_maskinit pixel map initial contour 시각화 옵션
    save_pixel_initial_contours = False  # pixel map에서 추출한 ct_score 필터링 전 initial contour 저장
    viz_pixel_initial_contours = False   # test 시 pixel initial contour 시각화
    viz_stage1_init = False              # Stage 1에서 poly_init 시각화 (ct_hm 또는 wh head 학습 시)
    single_rotate_angle = None
    use_rotate_tta = False
    reduce_step = 0.05
    reduce_min_vertices = 3
    reduce_apply_adaptive_th = False
    th_score_vertex_cls = 0.6
    use_vertex_reduction = True
    check_simple = False
    get_featuremap = False
    down_ratio = data.down_ratio
    task = commen.task
    vis_th_score = 0.3
    calc_deform_metric = False
    extract_offset = False
    vis_iou = False
    batch_size = 1
    test_stage = 'final'  # init, coarse final
    test_rescale = None
    ct_score = 0.05
    with_nms = True
    with_post_process = False
    segm_or_bbox = 'segm'
    dataset = 'sbd_val'

