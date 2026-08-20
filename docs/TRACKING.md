# Tracking Development

Two tracking approaches were implemented and evaluated.

## OC-SORT

The first tracking implementation used OC-SORT with the YOLO detector.

```text
YOLO
 ↓
Bounding Boxes
 ↓
OC-SORT
 ↓
Object IDs
```

Approximate performance:

**~30 FPS**

The implementation is available in:

```text
deployment_ocsort/
```

---

## Custom OpenCV Tracker

A custom OpenCV-based tracker was subsequently developed.

```text
YOLO
 ↓
Bounding Boxes
 ↓
Custom Tracker
 ↓
Object IDs
```

Approximate Python performance:

**~45 FPS**

The implementation is available in:

```text
deployment_custom_tracker/
```

---

## C++ Implementation

The custom tracker was then implemented in C++.

The overall tracking approach remained consistent while the runtime implementation
was moved from Python to C++.

Approximate performance:

**~60 FPS stable average**

The C++ implementation is available in:

```text
deployment_cpp/
```

## Tracking Objective

The tracking pipeline was designed to provide:

* Stable drone identities
* Consistent tracking across frames
* Real-time performance
* Low observed false-positive behavior

Performance values are specific to the project's Jetson Orin Nano test setup.
