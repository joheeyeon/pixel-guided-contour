# Contour-based Instance Segmentation with the Aid of Pixel-wise Classification and Topological Regularization

Official implementation of the IEEE Access paper  
**“Contour-based Instance Segmentation with the Aid of Pixel-wise Classification and Topological Regularization.”**

This repository provides the source code and dataset resources used in the paper.

---

## Overview

This work proposes a contour-based instance segmentation framework that improves contour evolution by incorporating pixel-wise classification guidance and topological regularization.
An auxiliary pixel classification head predicts dense interior–exterior score maps, which are fused with contour vertex features to guide iterative contour refinement.
In addition, geometric regularization terms are introduced to suppress self-intersections and enforce topological validity of predicted polygons.

---

## Repository Structure

```text
pixel-guided-contour/
 ├─ boundary-iou-api/     # Boundary IoU evaluation utilities
 ├─ configs/              # Configuration files
 ├─ data/                 # Dataset resources (annotations, splits, samples)
 ├─ dataset/              # Dataset loading and preprocessing code
 ├─ evaluator/            # Evaluation logic
 ├─ network/              # Model architectures and heads
 ├─ train/                # Training-related modules
 ├─ nms.py
 ├─ post_process.py
 ├─ train_net_lit.py
 ├─ test_lit.py
 ├─ requirements.txt
 ├─ LICENSE
 └─ README.md

## Dataset

This repository includes the dataset resources required to reproduce the experiments reported in the paper.

### PCB Bond Finger Dataset (industrial domain)

- COCO-format polygon annotations (JSON)
- Train/validation/test splits
- Representative sample images

### WHU Building Dataset (remote sensing domain)

- Train/validation/test splits
- Representative sample images

Due to confidentiality constraints, the full-resolution raw images of the PCB Bond Finger dataset are not publicly released.  
The provided annotations, splits, and sample images are sufficient to reproduce the reported results when combined with equivalent data.

Detailed dataset descriptions and formats are provided in `data/README.md`.

---

## Annotation Format

All annotations are provided in **COCO format** as JSON files.  
Object instances are annotated using polygon segmentations following the standard COCO specification.

---

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train_net_lit.py --config configs/pcb.yaml
```

## Evaluation

```bash
python test_lit.py --config configs/pcb.yaml
```

## License

The source code in this repository is released under the Apache License 2.0.

The dataset resources provided in this repository are intended for research purposes only and are not permitted for commercial use.

## Citation

This work has been submitted to IEEE Access.

If you find this repository useful, please consider citing the paper after publication.
