from .base import Dataset, RasterDataset
import os
import cv2
import numpy as np

class CocoDataset(Dataset):
    def process_info(self, ann):
        image_id = ann
        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=0)
        image_path = os.path.join(self.data_root, self.coco.loadImgs(int(image_id))[0]['file_name'])
        ann = self.coco.loadAnns(ann_ids)
        return ann, image_path, image_id

    def read_original_data(self, anno, image_path):
        img = cv2.imread(image_path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in instance['segmentation']] for instance in anno
                          if not isinstance(instance['segmentation'], dict)]
        cls_ids = [self.json_category_id_to_continuous_id[instance['category_id']] for instance in anno]
        return img, instance_polys, cls_ids

    def get_gt_img(self, image_id):
        if self.data_root_gt is not None:
            if self.coco.loadImgs(int(image_id))[0]['file_name'].split('_')[-1].split('.')[0] in ('lr','ud','udlr'):
                flip_type = self.coco.loadImgs(int(image_id))[0]['file_name'].split('_')[-1].split('.')[0]
                file_name = f"{'_'.join(self.coco.loadImgs(int(image_id))[0]['file_name'].split('_')[:-1])}.{self.coco.loadImgs(int(image_id))[0]['file_name'].split('_')[-1].split('.')[-1]}"
            else:
                flip_type = None
                file_name = self.coco.loadImgs(int(image_id))[0]['file_name']

            gt_image_path = os.path.join(self.data_root_gt, file_name)
        else:
            gt_image_path = None

        if gt_image_path is None:
            img_info = self.coco.imgs[image_id]
            gt_img = np.zeros((img_info['height'], img_info['width']))
        else:
            if os.path.isfile(gt_image_path):
                gt_img = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
            else:
                flag_load = False
                for ext in ['.png', '.jpg', '.TIF']:
                    if os.path.isfile(os.path.splitext(gt_image_path)[0] + ext):
                        gt_img = cv2.imread(os.path.splitext(gt_image_path)[0] + ext, cv2.IMREAD_GRAYSCALE)
                        flag_load = True
                        break
                if not flag_load:
                    img_info = self.coco.imgs[image_id]
                    gt_img = np.zeros((img_info['height'], img_info['width']))
            if flip_type == 'lr':
                gt_img = np.fliplr(gt_img)
            elif flip_type == 'ud':
                gt_img = np.flipud(gt_img)
            elif flip_type == 'udlr':
                gt_img = np.flipud(np.fliplr(gt_img))

        return gt_img
class CocoRasterDataset(RasterDataset):
    def process_info(self, ann):
        image_id = ann
        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=0)
        image_path = os.path.join(self.data_root, self.coco.loadImgs(int(image_id))[0]['file_name'])
        ann = self.coco.loadAnns(ann_ids)
        return ann, image_path, image_id

    def read_original_data(self, anno, image_path):
        img = cv2.imread(image_path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in instance['segmentation']] for instance in anno
                          if not isinstance(instance['segmentation'], dict)]
        cls_ids = [self.json_category_id_to_continuous_id[instance['category_id']] for instance in anno]
        return img, instance_polys, cls_ids