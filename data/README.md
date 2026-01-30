# Dataset Resources

This directory provides the dataset resources used to reproduce the experiments reported in the paper:

**“Contour-based Instance Segmentation with the Aid of Pixel-wise Classification and Topological Regularization.”**

The experiments are conducted on two datasets from different domains: an industrial PCB dataset and a public remote-sensing benchmark.

---

## 1. PCB Bond Finger Dataset (Industrial Domain)

The PCB Bond Finger dataset is collected from real-world industrial manufacturing lines.
It consists of cropped PCB images containing multiple gold-plated bond fingers located along the board edges.

### Resources provided in this repository

This repository provides the **complete PCB Bond Finger dataset**, including:

- Raw images
- COCO-format polygon annotations (`.json`)
- label images

The dataset is publicly released to support reproducibility and further research on high-precision contour-based instance segmentation and metrology-oriented applications.

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

### Usage in this work

For reproducibility, we provide:
- Train/validation/test split files used in our experiments

Users should download the full WHU Building Dataset from the official source and organize it according to the paths specified in the configuration files.

---

## 3. Notes on Usage

- The dataset resources provided in this repository are intended for **research purposes only**.
- Commercial use of the datasets is not permitted unless explicitly stated otherwise.
- Please refer to the main `README.md` for training and evaluation instructions.
