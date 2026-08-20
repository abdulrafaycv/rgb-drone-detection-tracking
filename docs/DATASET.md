# Dataset Development

## Base Dataset

The initial RGB drone detector was trained using a custom dataset.

The original base dataset cannot be publicly redistributed and is therefore not included in this repository.

To reproduce the training pipeline, users can substitute their own legally available RGB drone dataset using the YOLO annotation format.

### Preparation Pipeline

```text
RGB Images
    ↓
Annotation
    ↓
CVAT
    ↓
Label Validation
    ↓
Dataset Cleaning
    ↓
YOLO Format# Dataset Development

This project used custom RGB drone datasets for both the initial detector
training and subsequent fine-tuning.

The original datasets are not publicly redistributed in this repository.

---

## Base Dataset

The initial detector was trained using a custom RGB drone dataset.

The dataset contained annotated RGB images of drones and was prepared for
YOLO-based object detection.

The original dataset is not included because it cannot be publicly
redistributed.

Users can reproduce this stage using their own legally available RGB drone
dataset.

### Preparation Pipeline

```text
RGB Images
    ↓
Annotation
    ↓
CVAT
    ↓
Label Validation
    ↓
Dataset Cleaning
    ↓
YOLO Format
    ↓
Train / Validation Split
    ↓
Base Model Training
    ↓
Train / Validation Split
```

## Fine-Tuning Dataset

Additional data was collected to improve detector robustness under challenging
conditions.

The additional data included difficult examples such as severe motion blur.

The purpose of this stage was to expose the detector to conditions that were
less represented in the original training data.

## Annotation Format

The detector uses standard YOLO bounding-box annotations:

```text
class_id center_x center_y width height
```

Coordinates are normalized relative to image dimensions.

## Reproducing With Another Dataset

A compatible dataset should contain:

* RGB images
* Bounding-box annotations
* A consistent class definition
* Training/validation splits
* YOLO-compatible labels