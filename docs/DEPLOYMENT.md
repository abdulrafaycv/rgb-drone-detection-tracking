# Jetson Deployment

## Target Hardware

**NVIDIA Jetson Orin Nano**

The complete detection and tracking pipeline was optimized for real-time
edge deployment.

## Deployment Stages

### Stage 1 — OC-SORT

Python deployment using the fine-tuned YOLO detector, TensorRT/FP16 inference,
and OC-SORT.

**Approximate performance: ~30 FPS**

Location:

```text
deployment_ocsort/
```

---

### Stage 2 — Custom OpenCV Tracker

Python deployment using the fine-tuned detector and custom OpenCV tracker.

TensorRT/FP16 inference was used for optimized detector execution.

**Approximate performance: ~45 FPS**

Location:

```text
deployment_custom_tracker/
```

---

### Stage 3 — C++

The custom tracking pipeline was converted to C++.

The final implementation uses the TensorRT engine and custom OpenCV tracker.

**Approximate performance: ~60 FPS stable average**

Location:

```text
deployment_cpp/
```

### Running the C++ Deployment

Build the project:

```bash
./build.sh
```

Then launch:

```bash
./main1_cpp
```

## Environment

Record the actual deployment environment used for your experiments here:

```text
JetPack: 6.2
CUDA: 12.1
TensorRT: 10.3
OpenCV: 4.10
Python: 3.13
C++ Standard: c++17
Jetson Power Mode: 25 watt
Input Resolution: 640
Model Input Size: 640
```

Replace the placeholders with the actual values from your Jetson setup.

## Performance Considerations

Deployment performance depends on:

* TensorRT version
* CUDA version
* JetPack version
* Input resolution
* Model input size
* Jetson power mode
* Clock configuration
* Camera/video source
* Preprocessing and post-processing overhead
