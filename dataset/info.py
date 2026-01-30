
class DatasetInfo(object):
    dataset_info = {
        'whu_train': {
            'name': 'whu',
            'image_dir': 'data/whu/train',
            'anno_dir': 'data/whu/annotation/train.json',
            'gt_image_dir': 'data/whu/label/train',
            'split': 'train'
        },
        'whu_val': {
            'name': 'whu',
            'image_dir': 'data/whu/validation',
            'anno_dir': 'data/whu/annotation/validation.json',
            'gt_image_dir': 'data/whu/label/validation',
            'split': 'val'
        },
        'whu_test': {
            'name': 'whu',
            'image_dir': 'data/whu/test',
            'anno_dir': 'data/whu/annotation/test.json',
            'gt_image_dir': 'data/whu/label/test',
            'split': 'test'
        },
        'pcb_contour_test_all_v1': {
            'name': 'pcb',
            'image_dir': 'data/pcb/data_raster/pcb_remove_same_coarse_tv_raster_416_v4',
            'anno_file_list': ['coarse_pred_contour_e5', 'coarse_pred_contour_e15', 'coarse_pred_contour_e25',
                               'coarse_pred_contour_e35', 'coarse_pred_contour_e45', 'coarse_pred_contour_e55',
                               'coarse_pred_contour_e65', 'coarse_pred_contour_e75', 'coarse_pred_contour_e85',
                               'coarse_pred_contour_e95', 'coarse_pred_contour_e105', 'coarse_pred_contour_e115',
                               'coarse_pred_contour_e125', 'coarse_pred_contour_e135', 'coarse_pred_contour_e135',
                               'coarse_pred_contour_e145', 'coarse_pred_contour_e155', 'coarse_pred_contour_e165',
                               'coarse_pred_contour_e175'],
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'test'
        },
        'pcb_contour_val_all_v1': {
            'name': 'pcb',
            'image_dir': 'data/pcb/data_raster/pcb_remove_same_coarse_tv_raster_416_v4',
            'anno_file_list': ['coarse_pred_contour_e5', 'coarse_pred_contour_e25', 'coarse_pred_contour_e45', 'coarse_pred_contour_e85',
                               'coarse_pred_contour_e135', 'coarse_pred_contour_e180'],
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'val'
        },
        'pcb_contour_train_v1': {
            'name': 'pcb',
            'image_dir': 'data/pcb/data_raster/pcb_remove_same_coarse_tv_raster_416_v4',
            'anno_file_list': ['coarse_pred_contour_e5'],
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv4_0.2angle4_else1': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.2_else_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv4_0.2angle4_else1.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv6_0.15angle3_else2': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.15_else_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv6_0.15angle3_else2.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv5_0.2angle3_else1': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.2_else_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv5_0.2angle3_else1.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv4_0.15angle2_else1': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.15_else_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv4_0.15angle2_else1.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv4_0.2angle2_else1': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.2_else_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv4_0.2angle2_else1.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv4_0.15angle4': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.15_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv4_0.15angle4.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv4_0.2angle4': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.2_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv4_0.2angle4.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv4_0.15angle2': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.15_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv4_0.15angle2.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv4_0.2angle2': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.2_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv4_0.2angle2.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv4_0.15angle1': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.15_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv4_0.15angle1.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv4_0.2angle1': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.2_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv4_0.2angle1.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv_flip_v2_rep11': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_flip_v2',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv_flip_replicate11.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv_flip_v2_rep7': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_flip_v2',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv_flip_replicate7.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv_flip_v2_rep3': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_flip_v2',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv_flip_replicate3.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv_flip_v2': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_flip_v2',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv_flip_v2.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove_same_add_curv_flip': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv_flip.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_val_622_v2_remove_same': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_val_r2.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'val'
        },
        'pcb_train_622_v2_remove_same': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_test_622_v2_remove_same_reselect92human': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_test_r2_reselect92human.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'test'
        },
        'pcb_test_622_v2_remove_same_reselect90': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_test_r2_reselect90.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'test'
        },
        'pcb_test_622_v2_remove_same_reselect92': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_test_r2_reselect92.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'test'
        },
        'pcb_test_622_v2_remove_same_add_select_test': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_test_r2_add_select_test.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'test'
        },
        'pcb_test_622_v2_remove_same': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_test_r2.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'test'
        },
        'pcb_curved_test': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/instances_curved.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'test'
        },
        'pcb_curved': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/instances_curved.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_of4': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_train_r6_of4.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_of4_rest': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_train_r6_of4_rest.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_of8': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_train_r6_of8.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_of8_rest': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_train_r6_of8_rest.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_half': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_train_r3.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_half_rest': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_train_r3_rest.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_unsup': {
            'name': 'pcb',
            'image_dir': 'data/pcb/unsup/train/image',
            'anno_dir': 'data/pcb/annotations/instances_train_unsup_4of5.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove2': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_train_r6_remove2.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'train'
        },
        'pcb_train_622_v2_remove1': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_train_r6_remove1.json',
            'split': 'train'
        },
        'pcb_train_622_v2': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_train_r6.json',
            'split': 'train'
        },
        'pcb_test_622_v2': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_test_r2.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'test'
        },
        'pcb_train_622': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations/trn6val2tst2/instances_train_r6.json',
            'split': 'train'
        },
        'pcb_val_622': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations/trn6val2tst2/instances_val_r2.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'val'
        },
        'pcb_test_622': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations/trn6val2tst2/instances_test_r2.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'test'
        },
        'pcb_train': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations/instances_train_r8.json',
            'split': 'train'
        },
        'pcb_val': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations/instances_val_r2.json',
            'split': 'val'
        },
        'coco_train': {
            'name': 'coco',
            'image_dir': 'data/coco/train2017',
            'anno_dir': 'data/coco/annotations/instances_train2017.json',
            'split': 'train'
        },
        'coco_val': {
            'name': 'coco',
            'image_dir': 'data/coco/val2017',
            'anno_dir': 'data/coco/annotations/instances_val2017.json',
            'split': 'val'
        },
        'coco_test': {
            'name': 'coco',
            'image_dir': 'data/coco/test2017',
            'anno_dir': 'data/coco/annotations/image_info_test-dev2017.json',
            'split': 'test'
        },
        'sbd_train_of2': {
            'name': 'sbd',
            'image_dir': 'data/sbd/img',
            'anno_dir': 'data/sbd/annotations/sbd_train_1of2_sup_instance.json',
            'split': 'train'
        },
        'sbd_train': {
            'name': 'sbd',
            'image_dir': 'data/sbd/img',
            'anno_dir': 'data/sbd/annotations/sbd_train_instance.json',
            'split': 'train'
        },
        'sbd_val': {
            'name': 'sbd',
            'image_dir': 'data/sbd/img',
            'anno_dir': 'data/sbd/annotations/sbd_trainval_instance.json',
            'split': 'val'
        },
        'kitti_train': {
            'name': 'kitti',
            'image_dir': 'data/kitti/training/image_2', 
            'anno_dir': 'data/kitti/training/instances_train.json', 
            'split': 'train'
        }, 
        'kitti_val': {
            'name': 'kitti',
            'image_dir': 'data/kitti/testing/image_2', 
            'anno_dir': 'data/kitti/testing/instances_val.json', 
            'split': 'val'
        },
        'cityscapes_train': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit',
            'anno_dir': ('data/cityscapes/annotations/train', 'data/cityscapes/annotations/train_val'),
            'split': 'train'
        },
        'cityscapes_val': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit',
            'anno_dir': 'data/cityscapes/annotations/val',
            'split': 'val'
        },
        'cityscapesCoco_val': {
            'name': 'cityscapesCoco',
            'image_dir': 'data/cityscapes/leftImg8bit/val',
            'anno_dir': 'data/cityscapes/coco_ann/instance_val.json',
            'split': 'val'
        },
        'cityscapes_test': {
            'name': 'cityscapes',
            'image_dir': 'data/cityscapes/leftImg8bit/test', 
            'anno_dir': 'data/cityscapes/annotations/test', 
            'split': 'test'
        }
    }
