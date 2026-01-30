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
 │   ├─ setup.py
 │   ├─ README.md
 │   ├─ license.txt
 │   └─ init.txt
 ├─ configs/                   # Configuration files
 │   ├─ base.py
 │   ├─ pcb_remove_same_ccp_416_deform3_v104_norefine_notshare_softmax13_tl21_cv5.py
 │   └─ whu_ccp_512_softmax2_tl16_cv4.py
 ├─ data/                      # Dataset resources
 │   ├─ README.md
 │   ├─ pcb/
 │   └─ whu/
 ├─ dataset/                   # Dataset loading and preprocessing
 │   ├─ __init__.py
 │   ├─ collate_batch.py
 │   ├─ data_loader.py
 │   ├─ demo_dataset.py
 │   ├─ info.py
 │   ├─ test/                  # Test dataset classes
 │   │   ├─ base.py
 │   │   ├─ cityscapes.py
 │   │   ├─ cityscapesCoco.py
 │   │   ├─ coco.py
 │   │   ├─ kitti.py
 │   │   ├─ pcb.py
 │   │   ├─ sbd.py
 │   │   └─ whu.py
 │   └─ train/                 # Training dataset classes
 │       ├─ __init__.py
 │       ├─ base.py
 │       ├─ cityscapes.py
 │       ├─ cityscapesCoco.py
 │       ├─ coco.py
 │       ├─ douglas.py
 │       ├─ kitti.py
 │       ├─ pcb.py
 │       ├─ sbd.py
 │       ├─ utils.py
 │       └─ whu.py
 ├─ evaluator/                 # Evaluation metrics
 │   ├─ __init__.py
 │   ├─ make_evaluator.py
 │   ├─ pcb/                   # PCB dataset evaluation
 │   │   ├─ boundary.py
 │   │   ├─ cocoeval.py
 │   │   ├─ rasterize.py
 │   │   ├─ snake.py
 │   │   └─ utils.py
 │   └─ whu/                   # WHU dataset evaluation
 │       ├─ boundary.py
 │       ├─ cocoeval.py
 │       ├─ rasterize.py
 │       ├─ snake.py
 │       └─ utils.py
 ├─ network/                   # Model architectures
 │   ├─ __init__.py
 │   ├─ make_network.py
 │   ├─ data_utils.py
 │   ├─ extreme_utils_replacement.py
 │   ├─ backbone/              # Backbone networks
 │   │   ├─ dcn_v2.py
 │   │   ├─ dla.py
 │   │   └─ resnet.py
 │   ├─ detector_decode/       # Detection decoding heads
 │   │   ├─ refine_decode.py
 │   │   ├─ snake_decode.py
 │   │   ├─ utils.py
 │   │   └─ extreme_utils/     # CUDA utilities
 │   │       ├─ setup.py
 │   │       ├─ utils.cpp
 │   │       ├─ utils.h
 │   │       └─ src/
 │   │           ├─ cuda_common.h
 │   │           ├─ nms.cu
 │   │           ├─ nms.h
 │   │           └─ utils.cu
 │   └─ evolve/                # Contour evolution modules
 │       ├─ __init__.py
 │       ├─ evolve.py
 │       ├─ evolve_ccp.py
 │       ├─ evolve_rnn.py
 │       ├─ convlstm.py
 │       ├─ snake.py
 │       ├─ snake_evolve.py
 │       ├─ sharp.py
 │       └─ utils.py
 ├─ train/                     # Training modules
 │   ├─ __init__.py
 │   ├─ model_utils/
 │   │   └─ utils.py
 │   ├─ optimizer/
 │   │   └─ optimizer.py
 │   ├─ recorder/
 │   │   └─ recorder.py
 │   ├─ scheduler/
 │   │   └─ scheduler.py
 │   └─ trainer/               # Training logic for different methods
 │       ├─ make_trainer.py
 │       ├─ trainer.py
 │       ├─ ccp.py
 │       ├─ ccp_pyramid.py
 │       ├─ ccp_maskinit.py
 │       ├─ pixel.py
 │       ├─ coarse.py
 │       ├─ e2ec.py
 │       ├─ init.py
 │       ├─ rasterize.py
 │       ├─ deepsnake.py
 │       ├─ rnn.py
 │       ├─ snake_coarse.py
 │       ├─ snake_init.py
 │       └─ utils.py
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
