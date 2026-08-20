#!/usr/bin/env python3
import argparse
import sys
import os
import time
import queue
import threading
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Ensure local ocsort package is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ocsort import OCSort


def find_matching_det(trk_box, dets_info, iou_thresh=0.3):
    if not dets_info:
        return None
    tx1, ty1, tx2, ty2 = trk_box
    t_area = (tx2 - tx1) * (ty2 - ty1)
    if t_area <= 0:
        return None

    # Vectorized IoU calculation for maximum speed
    dx1 = np.array([d[0] for d in dets_info])
    dy1 = np.array([d[1] for d in dets_info])
    dx2 = np.array([d[2] for d in dets_info])
    dy2 = np.array([d[3] for d in dets_info])

    ix1 = np.maximum(tx1, dx1)
    iy1 = np.maximum(ty1, dy1)
    ix2 = np.minimum(tx2, dx2)
    iy2 = np.minimum(ty2, dy2)

    inter_w = np.maximum(0, ix2 - ix1)
    inter_h = np.maximum(0, iy2 - iy1)
    inter = inter_w * inter_h

    d_area = (dx2 - dx1) * (dy2 - dy1)
    union = t_area + d_area - inter

    valid = union > 0
    if not np.any(valid):
        return None

    ious = np.zeros(len(dets_info))
    ious[valid] = inter[valid] / union[valid]
    best_idx = np.argmax(ious)

    if ious[best_idx] >= iou_thresh:
        return (dets_info[best_idx][4], dets_info[best_idx][5])
    return None


def main():
    parser = argparse.ArgumentParser(description="Run High-FPS FP16 TensorRT YOLO live camera inference with OC-SORT tracking on Jetson Orin Nano.")
    parser.add_argument("--model", type=str, default="/home/dart/AbdulRafay/Week6/all_class/best.engine", help="Path to TensorRT .engine model")
    parser.add_argument("--precision", "--quantization", type=str, choices=["int8", "fp16", "fp32"], default="fp16", help="Model quantization/precision mode (int8, fp16, fp32)")
    parser.add_argument("--device", type=int, default=0, help="USB Camera device index (default: 0 for /dev/video0)")
    parser.add_argument("--fps", type=int, default=60, help="Target camera frame rate (default: 60 FPS)")
    parser.add_argument("--show", action="store_true", default=True, help="Display live video playback window")
    parser.add_argument("--no-show", action="store_false", dest="show", help="Disable display window (for maximum headless speed)")
    args = parser.parse_args()

    # 1. Check & Load TensorRT model
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Warning: Model at {model_path} not found. Searching directory...")
        fallback = "/home/dart/AbdulRafay/Week6/all_class/best.engine"
        if os.path.exists(fallback):
            model_path = fallback

    precision_mode = args.precision.upper()
    print(f"Loading TensorRT model [{precision_mode}]: {model_path}...")
    model = YOLO(model_path, task="detect")

    # Initialize OC-SORT Multi-Object Tracker (Tuned for anti-ID switching & 60 FPS tracking)
    tracker = OCSort(
        det_thresh=0.15,      # Capture low-conf detections to maintain track continuity
        max_age=90,           # Coast up to 90 frames (3s) to prevent track drops
        min_hits=1,           # Immediate track creation
        iou_threshold=0.05,   # Lower IoU threshold for small/fast moving boxes
        delta_t=3,
        asso_func="giou",     # GIoU (Generalized IoU) handles zero-overlap fast moving boxes
        inertia=0.3,          # Increased trajectory inertia for smoother prediction
        use_byte=True,        # BYTE-style secondary association for low-conf detections
    )

    use_half = (args.precision == "fp16")

    # 2. Camera Device Discovery & Check
    target_device = args.device
    dev_path = f"/dev/video{target_device}"

    if not os.path.exists(dev_path):
        # Auto-discover available video devices on Linux
        available_devices = [d for d in range(10) if os.path.exists(f"/dev/video{d}")]
        if available_devices:
            target_device = available_devices[0]
            dev_path = f"/dev/video{target_device}"
            print(f"Notice: /dev/video{args.device} not found. Auto-detected available camera at {dev_path}.")
        else:
            print("\n" + "=" * 70)
            print(" ERROR: No video capture devices found at /dev/video*")
            print("=" * 70)
            print(" 1. Please plug in a USB camera or CSI camera module.")
            print(" 2. If using a video file instead, run:")
            print("    python run_video_engine.py --video bw_video.mp4")
            print("=" * 70 + "\n")
            return

    # 3. NVIDIA DeepStream Hardware Accelerated Camera Pipeline (Configured for 60 FPS)
    ds_cam_mjpeg = (
        f"v4l2src device={dev_path} ! "
        f"image/jpeg, width=640, height=480, framerate={args.fps}/1 ! "
        f"nvjpegdec ! video/x-raw(memory:NVMM) ! "
        f"nvvidconv ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )

    ds_cam_raw = (
        f"v4l2src device={dev_path} ! "
        f"video/x-raw, width=640, height=480, framerate={args.fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )

    print(f"Attempting to open camera {dev_path} using NVIDIA DeepStream 60 FPS pipeline...")
    cap = cv2.VideoCapture(ds_cam_mjpeg, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(ds_cam_raw, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"DeepStream HW pipeline unavailable. Using V4L2 VideoCapture (60 FPS MJPG) for {dev_path}...")
        cap = cv2.VideoCapture(target_device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        print(f"Successfully connected to camera using NVIDIA DeepStream HW Decoder ({args.fps} FPS)!")

    if not cap.isOpened():
        print(f"Error: Could not open USB camera on {dev_path}. Please check camera connection.")
        return

    # 3. Asynchronous Multi-Threaded Camera Frame Reader (Drop oldest frames to minimize latency)
    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    def camera_reader_thread():
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                frame_queue.put(None)
                break
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)

    t_reader = threading.Thread(target=camera_reader_thread, daemon=True)
    t_reader.start()

    prev_time = time.time()
    fps = 0.0
    locked_id = None         # Persistent target lock ID
    lock_lost_frames = 0     # Grace period counter for brief detection dropouts
    target_width = 640
    target_height = 640

    print(f"Starting 60 FPS live camera processing with OC-SORT [{precision_mode}]. Press 'q' in the window to stop.")

    # 4. Main 60 FPS High-Speed Processing Loop
    while True:
        try:
            frame = frame_queue.get(timeout=2.0)
        except queue.Empty:
            break

        if frame is None:
            print("Camera feed stopped.")
            break

        t_start = time.time()

        # Fast CPU Resize if needed
        if frame.shape[0] != target_height or frame.shape[1] != target_width:
            input_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        else:
            input_frame = frame

        # High-Performance Direct TensorRT Inference
        results = model(input_frame, conf=0.15, verbose=False, half=use_half)
        latency_ms = (time.time() - t_start) * 1000.0

        # Extract YOLO detections for OC-SORT tracker
        det_list = []
        det_info_list = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes
            xyxys = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(confs)):
                c_id = clss[i]
                c_name = model.names[c_id] if hasattr(model, 'names') and c_id in model.names else str(c_id)
                det_list.append([xyxys[i, 0], xyxys[i, 1], xyxys[i, 2], xyxys[i, 3], confs[i]])
                det_info_list.append((xyxys[i, 0], xyxys[i, 1], xyxys[i, 2], xyxys[i, 3], confs[i], c_name))

        if len(det_list) > 0:
            dets = np.array(det_list, dtype=np.float32)
        else:
            dets = np.empty((0, 5), dtype=np.float32)

        # Update OC-SORT Tracker
        tracks = tracker.update(dets, (target_height, target_width), (target_height, target_width))

        # Persistent Target Lock-On with Dropout Memory Buffer (Anti-ID Switching)
        active_ids = [int(t[4]) for t in tracks] if len(tracks) > 0 else []
        if locked_id is not None and locked_id in active_ids:
            lock_lost_frames = 0
        elif locked_id is not None and locked_id not in active_ids:
            lock_lost_frames += 1
            if lock_lost_frames > 30:  # 30 frames grace period
                locked_id = min(active_ids) if len(active_ids) > 0 else None
                lock_lost_frames = 0
        elif len(active_ids) > 0:
            locked_id = min(active_ids)
            lock_lost_frames = 0
        else:
            locked_id = None
            lock_lost_frames = 0

        # Fast direct tracked bounding box rendering
        annotated_frame = input_frame
        box_sizes = []
        for trk in tracks:
            tx1, ty1, tx2, ty2, track_id = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3]), int(trk[4])
            bw = tx2 - tx1
            bh = ty2 - ty1
            box_sizes.append((bw, bh))

            is_locked = (track_id == locked_id)
            matched_info = find_matching_det((tx1, ty1, tx2, ty2), det_info_list)

            if is_locked:
                if matched_info:
                    conf, cls_name = matched_info
                    id_label = f"LOCKED | OC-SORT ID: #{track_id} ({cls_name} {conf:.2f})"
                else:
                    id_label = f"LOCKED | OC-SORT ID: #{track_id}"
                color = (0, 255, 0)
                thickness = 3
            else:
                if matched_info:
                    conf, cls_name = matched_info
                    id_label = f"OC-SORT ID: #{track_id} ({cls_name} {conf:.2f})"
                else:
                    id_label = f"OC-SORT ID: #{track_id}"
                color = (255, 165, 0)
                thickness = 2

            # Bounding box
            cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), color, thickness)

            # Prominent filled background for OC-SORT ID label for maximum legibility
            font_scale = 0.55 if is_locked else 0.50
            font_thick = 2 if is_locked else 1
            label_y = max(ty1 - 8, 18)
            cv2.rectangle(annotated_frame, (tx1, label_y - 14), (tx1 + len(id_label) * 9, label_y + 4), (0, 0, 0), -1)
            cv2.putText(annotated_frame, id_label, (tx1 + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA)

            # Size text below box
            size_text = f"Size: {bw}x{bh} px"
            cv2.putText(annotated_frame, size_text, (tx1, min(target_height - 5, ty2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Calculate FPS with exponential smoothing
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps = 0.8 * fps + 0.2 * inst_fps if fps > 0 else inst_fps

        # Draw HUD overlay (FPS, Latency, Precision, Active Tracks, Target Lock Status, Box Size)
        lock_status_text = f"LOCKED (ID: {locked_id})" if locked_id is not None else "SEARCHING"
        cv2.putText(annotated_frame, f"FPS: {fps:.1f} (Target: {args.fps} FPS) [{precision_mode}] | Tracks: {len(tracks)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Target Status: {lock_status_text}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if locked_id else (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Latency: {latency_ms:.1f} ms", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        if box_sizes:
            sizes_str = ", ".join([f"{w}x{h}px" for w, h in box_sizes[:3]])
            cv2.putText(annotated_frame, f"BBox Size: {sizes_str}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Display live camera window
        if args.show:
            cv2.imshow(f"Jetson Orin Nano - Live TensorRT DeepStream Detection [{precision_mode}]", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Interrupted by user.")
                stop_event.set()
                break

    # Clean up
    stop_event.set()
    if t_reader.is_alive():
        t_reader.join(timeout=1.0)
    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
