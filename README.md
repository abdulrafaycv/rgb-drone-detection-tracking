# RGB Drone Detection, Tracking & Deployment

An end-to-end real-time RGB drone detection and multi-object tracking system, developed from custom dataset creation and model training through optimized deployment on an NVIDIA Jetson Orin Nano.

The project progressed from a YOLO-based drone detector to multiple real-time tracking implementations and a final C++ deployment pipeline.

---

## Project Overview

The system was developed through the following stages:

1. Created and annotated a custom RGB drone dataset.
2. Trained a base YOLO detector.
3. Collected additional data containing challenging conditions such as severe motion blur.
4. Fine-tuned the detector on the expanded dataset.
5. Integrated the detector with OC-SORT for multi-object tracking.
6. Developed a custom OpenCV-based tracking solution.
7. Optimized inference using TensorRT and FP16.
8. Converted the custom tracking pipeline from Python to C++.
9. Deployed the final system on an NVIDIA Jetson Orin Nano.

---

## System Pipeline

```text
Custom RGB Dataset
        │
        ▼
   Base YOLO Model
        │
        ▼
Additional Challenging Data
   (Motion Blur, etc.)
        │
        ▼
 Fine-Tuned YOLO Model
        │
        ├───────────────┐
        ▼               ▼
    OC-SORT       Custom OpenCV Tracker
        │               │
        ▼               ▼
     ~30 FPS          ~45 FPS
                        │
                        ▼
                   C++ Version
                        │
                        ▼
                     ~60 FPS
                        │
                        ▼
              Jetson Orin Nano
```

---

## Detection

### Base Model

A custom RGB drone dataset was used to train the initial YOLO detector.

The base model achieved:

* **97.0% mAP50**
* **64.6% mAP50-95**

The original base dataset is not included in this repository because it cannot be publicly redistributed.

The repository instead provides the training configuration and model artifacts. Users can reproduce the training stage using their own appropriately annotated drone dataset or another legally redistributable drone dataset.

See [`datasets/README.md`](datasets/README.md).

---

### Fine-Tuned Model

Additional RGB data was collected to expose the detector to more challenging conditions, including severe motion blur.

The resulting fine-tuned detector achieved:

* **94.5% mAP50**
* **62.2% mAP50-95**

The original dataset is not included in this repository because it cannot be
publicly redistributed.

See [`datasets/README.md`](datasets/README.md).

---

## Tracking & Deployment

Three implementations are included in this repository.

### 1. YOLO + OC-SORT

The fine-tuned detector was integrated with OC-SORT for multi-object tracking.

The deployment uses:

* TensorRT
* FP16 inference
* NVIDIA Jetson Orin Nano

Approximate project performance:

**~30 FPS**

See [`deployment_ocsort/`](deployment_ocsort/).

---

### 2. YOLO + Custom OpenCV Tracker

A custom OpenCV-based tracker was developed and integrated with the fine-tuned detector.

The folder contains both the tracker and the deployment implementation.

The deployment supports the project's TensorRT/FP16 inference pipeline.

Approximate project performance:

**~45 FPS**

See [`deployment_custom_tracker/`](deployment_custom_tracker/).

---

### 3. C++ + Custom OpenCV Tracker

The custom tracking implementation was converted from Python to C++ for improved real-time performance.

The C++ implementation uses the same general detection and tracking approach while reducing Python runtime overhead.

Approximate project performance:

**~60 FPS stable average**

See [`deployment_cpp/`](deployment_cpp/).

---

## Performance

| Implementation        | Language | Tracking      | Approx. FPS |
| --------------------- | -------- | ------------- | ----------: |
| YOLO + OC-SORT        | Python   | OC-SORT       |     ~30 FPS |
| YOLO + Custom Tracker | Python   | Custom OpenCV |     ~45 FPS |
| YOLO + Custom Tracker | C++      | Custom OpenCV |     ~60 FPS |

> FPS values are project-specific measurements obtained on the NVIDIA Jetson Orin Nano under the project's test configuration. They should not be interpreted as universal benchmarks.

---

## Hardware

**NVIDIA Jetson Orin Nano**

The detection and tracking pipelines were optimized for real-time edge inference on the Jetson platform.

---

## Technologies

* Ultralytics YOLO
* PyTorch
* Python
* C++
* OpenCV
* CVAT
* OC-SORT
* TensorRT
* FP16 inference
* ONNX
* NVIDIA Jetson Orin Nano

---

## Repository Structure

```text
├── datasets/
│   └── README.md

├── training/
│   ├── base/
│   └── finetuned/

├── deployment_ocsort/
│   ├── oc_sort/
│   ├── best.engine
│   └── run_live_engine.py

├── deployment_custom_tracker/
│   ├── drone_tracker.py
│   ├── main1.py
│   ├── rgb_finetuned.pt
│   └── rgb_finetuned.engine

├── deployment_cpp/
│   ├── build.sh
│   ├── drone_tracker.cpp
│   ├── drone_tracker.hpp
│   ├── main1.cpp
│   ├── main1_cpp
│   ├── rgb_finetuned.pt
│   ├── rgb_finetuned.onnx
│   └── rgb_finetuned.engine

├── results/
└── docs/
```

---

## Reproducibility

The repository contains training configurations, model weights, tracking implementations, deployment scripts, and compiled/deployment artifacts used during development.

Exact results may vary depending on:

* Dataset and train/validation split
* YOLO model configuration
* Input resolution
* TensorRT version
* CUDA/JetPack version
* OpenCV version
* Jetson power mode
* Input stream resolution
* Runtime configuration

For detailed information, see the documentation in [`docs/`](docs/).

---

## Project Result

The project evolved from a standalone RGB drone detector into a complete real-time detection, tracking, and edge-deployment pipeline.

**Final reported performance: ~60 FPS on NVIDIA Jetson Orin Nano using the C++ implementation.**
