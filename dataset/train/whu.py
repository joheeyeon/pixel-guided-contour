if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    from train.coco import CocoDataset, CocoRasterDataset
else:
    from .coco import CocoDataset, CocoRasterDataset

class WhuDataset(CocoDataset):
    pass

class WhuRasterDataset(CocoRasterDataset):
    pass


if __name__ == "__main__":
    import importlib, cv2, os
    import numpy as np
    data_info = {
        'name': 'whu',
        'image_dir': 'data/whu/train/image_aug_curve_angle0.2_flip',
        'anno_dir': 'data/whu/annotations_v2/trn6val2tst2/instances_remove_same_train_r6_add_curv4_0.2angle4.json',
        'split': 'train'
    }
    sys.path.insert(0, "/data/HY/Code/Segmentation/Instance/Snake-based/my-contour-snake")
    cfg = importlib.import_module('configs.whu.e2ec+pix.pcb_remove_same_e2ec+inpix+jointpix_416_v6')
    dataset = PcbDataset(data_info['anno_dir'], data_info['image_dir'], data_info['split'], cfg, data_info['gt_image_dir'] if 'gt_image_dir' in data_info else None)
    directory_path = "data/analyze/pcb_train_622_v2_remove_same_add_curv4_0.2angle4/data.scale=None" #,augment_shift=False
    os.makedirs(directory_path, exist_ok=True)
    for i in range(len(dataset)):
        batch = dataset.__getitem__(i)
        img = batch['orig_img']
        gt_mask = batch['pixel_gt']
        gt_mask = np.stack([gt_mask, gt_mask, gt_mask],axis=-1)
        cv2.imwrite(f"{directory_path}/{i}_{dataset.coco.loadImgs(int(batch['meta']['img_id']))[0]['file_name']}", np.concatenate((img, gt_mask),1))