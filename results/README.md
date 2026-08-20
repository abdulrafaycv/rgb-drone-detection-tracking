# Project Results

This directory contains project-level results and performance comparisons.

Detailed training metrics and evaluation plots are stored with their respective
training runs:

```text
training/base/results/
training/finetuned/results/
```

## Detection Results

### Base Detector

* mAP50: **97.0%**
* mAP50-95: **64.6%**

### Fine-Tuned Detector

* mAP50: **94.5%**
* mAP50-95: **62.2%**

The fine-tuned model was evaluated under the corresponding project validation
conditions, which included more challenging imagery.

## Tracking & Deployment

| Pipeline                     | Approx. FPS |
| ---------------------------- | ----------: |
| YOLO + OC-SORT               |     ~30 FPS |
| YOLO + Custom OpenCV Tracker |     ~45 FPS |
| C++ + Custom OpenCV Tracker  |     ~60 FPS |

These are project-specific measurements obtained on NVIDIA Jetson Orin Nano.
