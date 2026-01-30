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
```

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
