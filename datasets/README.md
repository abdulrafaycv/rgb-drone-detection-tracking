# Dataset

This project uses two stages of RGB drone data:

1. Base training data
2. Fine-tuning data

The original datasets are not included in this repository. The training
configuration and pretrained model weights are provided so that the pipeline
can be reproduced using a suitable dataset.

---

## 1. Base Training Dataset

The initial YOLO detector was trained on a custom RGB drone dataset.

The original dataset is not included because it cannot be publicly
redistributed.

### Reproducing the Base Training

Users can create their own base dataset using legally available RGB drone
imagery.

The dataset should contain:

- RGB images
- Drone bounding-box annotations
- Consistent class definitions
- Training and validation splits
- YOLO-format labels

A recommended structure is:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── data.yaml

The dataset can be created and annotated using CVAT or another annotation
tool that supports YOLO-format export.

The training configuration used for the project is provided in:

training/base/

The base model weights are also provided, allowing users to directly use the
trained detector without reproducing the original training dataset.

## 2. Fine-Tuning Dataset

The fine-tuning stage was designed to improve the robustness of the detector
under more difficult real-world conditions.

The original fine-tuning dataset is not included in this repository.

Instead, users can create their own challenging dataset and reproduce the
fine-tuning stage using the provided training configuration and pretrained
weights.

What Should the Fine-Tuning Data Contain?

The additional data should focus on conditions that are difficult for small
drone detection.

Examples include:

Severe motion blur
Fast drone movement
Camera movement
Different lighting conditions
Very bright scenes
Low-light and dark scenes
High dynamic range scenes
Strong shadows
Backlighting
Overexposure
Underexposure
Low-contrast scenes
Small and distant drones
Partial occlusion
Cluttered backgrounds
Complex backgrounds
Different weather conditions
Different viewing angles
Different drone orientations
Different image resolutions
Compression artifacts
Other extreme or uncommon conditions encountered in real-world operation

The objective is not simply to collect more images, but to collect images that
represent failure cases and difficult conditions that are underrepresented in
the original training data.

Fine-Tuning Workflow

A recommended workflow is:

Base Drone Dataset
        ↓
Train Base YOLO Model
        ↓
Evaluate Base Model
        ↓
Identify Difficult / Failure Cases
        ↓
Collect Additional Challenging Data
        ↓
Annotate & Validate
        ↓
Combine / Prepare Fine-Tuning Dataset
        ↓
Fine-Tune Base Model
        ↓
Evaluate Fine-Tuned Model

The fine-tuning configuration is available in:

training/finetuned/

The pretrained model weights required to start the fine-tuning process are
provided in the training directory.

Using the Provided Weights Instead

Recreating the exact datasets used during development is not required to use
the trained models.

Pretrained model weights are provided in:

training/base/
training/finetuned/

The available best.pt weights can be used directly for inference or as the
starting point for further fine-tuning on a user's own dataset.

This allows the project to be used even when the original datasets are not
available.

Dataset Preparation

When creating a new dataset, ensure that:

Images are correctly annotated.
Bounding boxes tightly cover the visible drone.
Labels use the correct class IDs.
Training and validation data are separated appropriately.
Duplicate or corrupted images are removed.
Labels are validated before training.
Challenging scenarios are sufficiently represented.

Dataset quality is particularly important for small-object drone detection,
where a small labeling error can have a significant effect on training and
evaluation.