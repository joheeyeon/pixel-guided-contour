# Dataset Resources

This directory provides the dataset resources used to reproduce the experiments reported in the paper:

**“Contour-based Instance Segmentation with the Aid of Pixel-wise Classification and Topological Regularization.”**

The experiments are conducted on two datasets from different domains: an industrial PCB dataset and a public remote-sensing benchmark.

---

## 1. PCB Bond Finger Dataset (Industrial Domain)

The PCB Bond Finger dataset is collected from real-world industrial manufacturing lines.
It consists of cropped PCB images containing multiple gold-plated bond fingers located along the board edges.
The full PCB Bond Finger dataset (images and annotations) is publicly available via Google Drive:

https://drive.google.com/drive/folders/1kub2J6OgIgowm8CEfyQgzCngzu3bYZKZ?usp=sharing

Due to the file size limitations of GitHub, the dataset is hosted externally.
Users should download the dataset from the link above and place it under the appropriate data directory before training or evaluation.

### Resources provided in this repository

This repository provides the **complete PCB Bond Finger dataset**, including:

- Raw images
- COCO-format polygon annotations (`.json`)
- label images

The dataset is publicly released to support reproducibility and further research on high-precision contour-based instance segmentation and metrology-oriented applications.

### Directory Structure

After downloading, organize the PCB dataset as follows:

```text
data/pcb/
 ├─ train/
 │   ├─ image/                          # Original training and testing images
 │   ├─ image_aug_curve_angle0.2_flip/  # Augmented training images
 │   └─ label_v2/                       # Ground truth label images
 └─ annotations_v2/
     └─ trn6val2tst2/
         ├─ instances_remove_same_train_r6_add_curv4_0.2angle4.json
         ├─ instances_remove_same_val_r2.json
         └─ instances_remove_same_test_r2_add_select_test.json
```

### Dataset Keys

The following dataset keys are defined in `dataset/info.py` for training/validation/testing:

- **`pcb_train_622_v2_remove_same_add_curv4_0.2angle4`** - Training set (uses augmented images)
- **`pcb_val_622_v2_remove_same`** - Validation set
- **`pcb_test_622_v2_remove_same_add_select_test`** - Test set

### Annotation format

All annotations are provided in **COCO format** as JSON files.
Each object instance is represented by a polygon segmentation defined as an ordered list of vertices following the standard COCO specification.

---

## 2. WHU Building Dataset (Remote Sensing Domain)

The WHU Building Dataset is a publicly available benchmark for large-scale building extraction from aerial imagery.

In this study, we follow the official dataset definition and evaluation protocol provided by the dataset authors.
No image data from the WHU Building Dataset is redistributed in this repository.

### Official dataset source

The full dataset, including images and annotations, can be obtained from the official website:

- WHU Building Dataset: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html

### Directory Structure

After downloading from the official source, organize the WHU dataset as follows:

```text
data/whu/
 ├─ train/             # Training images
 ├─ validation/        # Validation images
 ├─ test/              # Test images
 ├─ label/
 │   ├─ train/         # Training label images
 │   ├─ validation/     # Validation label images
 │   └─ test/          # Test label images
 └─ annotation/
     ├─ train.json     # Training annotations (COCO format)
     ├─ validation.json # Validation annotations (COCO format)
     └─ test.json      # Test annotations (COCO format)
```

### Dataset Keys

The following dataset keys are defined in `dataset/info.py` for training/validation/testing:

- **`whu_train`** - Training set
- **`whu_val`** - Validation set
- **`whu_test`** - Test set

### Usage in this work

For reproducibility, we provide:
- Train/validation/test split files used in our experiments

Users should download the full WHU Building Dataset from the official source and organize it according to the paths specified in the configuration files.

---

## 3. Notes on Usage

- The dataset resources provided in this repository are intended for **research purposes only**.
- Commercial use of the datasets is not permitted unless explicitly stated otherwise.
- **Directory structure must match the paths specified in `dataset/info.py`** for correct data loading.
- Training uses augmented images while validation and testing use original images (see PCB dataset structure).
- Refer to the main `README.md` for training and evaluation instructions.
- The dataset keys listed above should be used in training/testing configuration files.
