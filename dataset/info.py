
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
        'pcb_train_622_v2_remove_same_add_curv4_0.2angle4': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image_aug_curve_angle0.2_flip',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv4_0.2angle4.json',
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

        'pcb_test_622_v2_remove_same_add_select_test': {
            'name': 'pcb',
            'image_dir': 'data/pcb/train/image',
            'anno_dir': 'data/pcb/annotations_v2/trn6val2tst2/instances_remove_same_test_r2_add_select_test.json',
            'gt_image_dir': 'data/pcb/train/label_v2',
            'split': 'test'
        }
    }
