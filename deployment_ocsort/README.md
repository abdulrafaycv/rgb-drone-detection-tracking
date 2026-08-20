# YOLO + OC-SORT Deployment

Real-time RGB drone detection and multi-object tracking using the fine-tuned YOLO detector and OC-SORT.

## Contents

```text
deployment_ocsort/
├── oc_sort/
│   └── OC-SORT implementation files
├── best.engine
├── run_live_engine.py
└── README.md
```

### `oc_sort/`

Contains the OC-SORT implementation used for multi-object tracking.

### `best.engine`

TensorRT engine generated from the trained YOLO detector for optimized inference.

The engine is configured for the project's FP16 deployment pipeline.

### `run_live_engine.py`

Main live inference and tracking script.

The script combines:

```text
TensorRT YOLO Detection
        ↓
Detection Post-Processing
        ↓
OC-SORT
        ↓
Tracked Drone Output
```

## Running

Run the live deployment script using the appropriate Python environment on the Jetson Orin Nano:

```bash
python3 run_live_engine.py
```

Make sure the required TensorRT engine, OC-SORT files, Python dependencies, and camera/video input configuration are available.

## Hardware

* NVIDIA Jetson Orin Nano

## Inference

* TensorRT
* FP16
* YOLO
* OC-SORT

## Performance

Approximate performance observed during project testing:

**~30 FPS**

> Performance depends on the input resolution, model configuration, TensorRT/CUDA environment, Jetson power mode, and input source.

## Notes

The provided TensorRT engine is environment-dependent. If the engine cannot be used directly on another Jetson/software configuration, regenerate the engine from the compatible model and deployment configuration.
