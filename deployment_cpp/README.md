# C++ YOLO + Custom OpenCV Tracker

C++ implementation of the real-time RGB drone detection and custom tracking pipeline.

This implementation converts the Python-based custom tracker pipeline to C++ for improved runtime performance on the NVIDIA Jetson Orin Nano.

## Contents

```text
deployment_cpp/
├── build.sh
├── drone_tracker.cpp
├── drone_tracker.hpp
├── main1.cpp
├── main1_cpp
├── rgb_finetuned.pt
├── rgb_finetuned.onnx
├── rgb_finetuned.engine
└── README.md
```

### Source Files

**`drone_tracker.cpp` / `drone_tracker.hpp`**

Custom OpenCV-based tracking implementation.

**`main1.cpp`**

Main C++ application and deployment pipeline.

**`build.sh`**

Build script used to compile the C++ implementation.

**`main1_cpp`**

Compiled executable used to run the deployment.

### Model Files

**`rgb_finetuned.pt`**

Fine-tuned YOLO model weights.

**`rgb_finetuned.onnx`**

ONNX representation of the fine-tuned detector.

**`rgb_finetuned.engine`**

TensorRT engine used for optimized inference.

## Pipeline

```text
RGB Input
    ↓
TensorRT YOLO Detection
    ↓
Detection Post-Processing
    ↓
Custom OpenCV Tracker
    ↓
Tracked Drone Output
```

## Build

Make the build script executable if necessary:

```bash
chmod +x build.sh
```

Then run:

```bash
./build.sh
```

This builds the C++ deployment executable.

## Run

After building the project, run:

```bash
./main1_cpp
```

The executable launches the C++ detection and tracking pipeline.

## Hardware

* NVIDIA Jetson Orin Nano

## Optimization

The C++ implementation was developed to reduce Python runtime overhead and improve real-time performance.

The deployment uses:

* C++
* OpenCV
* TensorRT
* FP16 inference
* NVIDIA Jetson Orin Nano

## Performance

Approximate project performance:

**~60 FPS stable average**

The final implementation provided stable real-time tracking with a low observed false-positive rate under the project's test conditions.

> FPS is a project-specific measurement and can vary with model configuration, input resolution, TensorRT/CUDA versions, Jetson power mode, and input source.

## Important

TensorRT engine files can be environment-dependent. If `rgb_finetuned.engine` is incompatible with the target environment, regenerate the engine from the compatible ONNX/model configuration.
