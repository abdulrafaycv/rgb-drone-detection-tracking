# YOLO + Custom OpenCV Tracker

Real-time RGB drone detection and tracking using the fine-tuned YOLO detector and a custom OpenCV-based tracking implementation.

## Contents

```text
deployment_custom_tracker/
├── drone_tracker.py
├── main1.py
├── rgb_finetuned.pt
├── rgb_finetuned.engine
└── README.md
```

### `drone_tracker.py`

Contains the custom OpenCV-based tracking implementation.

### `main1.py`

Main deployment script.

The implementation supports the project's detection/tracking workflow and can operate in the available detection or tracking modes.

### `rgb_finetuned.pt`

Fine-tuned YOLO model weights.

### `rgb_finetuned.engine`

TensorRT engine generated from the fine-tuned model for optimized inference.

## Pipeline

```text
RGB Input
   ↓
YOLO Detection
   ↓
Detection / Tracking Mode
   ↓
Custom OpenCV Tracker
   ↓
Tracked Drone Output
```

## Running

Run the deployment script using the configured Python environment:

```bash
python3 main1.py
```

Make sure the required model files, TensorRT environment, OpenCV installation, and input source are correctly configured.

## Hardware

* NVIDIA Jetson Orin Nano

## Optimization

* TensorRT
* FP16 inference
* OpenCV

## Performance

Approximate project performance:

**~45 FPS**

> FPS is a project-specific measurement on the NVIDIA Jetson Orin Nano and may vary with input resolution, model configuration, TensorRT/CUDA versions, power mode, and input source.

## Notes

The `.engine` file is hardware/software-environment dependent. If it cannot be loaded in another environment, regenerate the TensorRT engine from the compatible model.
