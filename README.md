# Contour-based Instance Segmentation with the Aid of Pixel-wise Classification and Topological Regularization

Official implementation of a paper submitted to IEEE Access  
“Contour-based Instance Segmentation with the Aid of Pixel-wise Classification and Topological Regularization.”

This repository provides the source code and dataset resources used in the submitted manuscript.

---

## Overview

This work proposes a contour-based instance segmentation framework that improves contour evolution by incorporating pixel-wise classification guidance and topological regularization.
An auxiliary pixel classification head predicts dense interior–exterior score maps, which are fused with contour vertex features to guide iterative contour refinement.
In addition, geometric regularization terms are introduced to suppress self-intersections and enforce topological validity of predicted polygons.

---

## Repository Structure

```text
pixel-guided-contour/
 ├─ boundary-iou-api/          # Boundary IoU evaluation utilities
 ├─ configs/                   # Configuration files
 ├─ data/                      # Dataset resources (PCB, WHU)
 ├─ dataset/                   # Dataset loading and preprocessing
 │   ├─ test/                  # Test dataset classes
 │   └─ train/                 # Training dataset classes
 ├─ evaluator/                 # Evaluation metrics
 │   ├─ pcb/                   # PCB dataset evaluation
 │   └─ whu/                   # WHU dataset evaluation
 ├─ network/                   # Model architectures
 │   ├─ backbone/              # Backbone networks (DLA, ResNet)
 │   ├─ detector_decode/       # Detection decoding heads
 │   └─ evolve/                # Contour evolution modules
 ├─ train/                     # Training modules
 │   ├─ model_utils/           # Model utilities
 │   ├─ optimizer/             # Optimizer configuration
 │   ├─ recorder/              # Training recorder
 │   ├─ scheduler/             # Learning rate scheduler
 │   └─ trainer/               # Training logic for different methods
 ├─ train_net_lit.py          # Training script
 ├─ test_lit.py               # Testing script
 ├─ requirements.txt
 ├─ LICENSE
 └─ README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train_net_lit.py pcb_remove_same_ccp_416_deform3_v104_norefine_notshare_softmax13_tl21_cv5 --exp exp0 --epochs 100
```

## Evaluation

```bash
python test_lit.py pcb_remove_same_ccp_416_deform3_v104_norefine_notshare_softmax13_tl21_cv5 --checkpoint /path/to/checkpoint.pth --viz
```

### Common Test Options

- `--with_nms`: Apply Non-Maximum Suppression
- `--nms_iou_th 0.5`: IoU threshold for NMS (default: 0.5)
- `--nms_containment_th 0.7`: Containment threshold for NMS (default: 0.7)
- `--eval segm`: Evaluate segmentation (default) or `bbox` for bounding box
- `--ct_score 0.05`: Confidence threshold for contour filtering (default: 0.05)
- `--viz_mode final`: Visualization mode - `final` (last stage only) or `timeline` (all stages)

## License

The source code in this repository is released under the Apache License 2.0.

The dataset resources provided in this repository are intended for research purposes only and are not permitted for commercial use.

## Citation

This work has been submitted to IEEE Access.

If you find this repository useful, please consider citing the paper after publication.
