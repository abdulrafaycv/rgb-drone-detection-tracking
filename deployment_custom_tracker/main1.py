import cv2
import time
import os
import threading
import queue
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from drone_tracker import DroneTracker

# Try importing TensorRT for zero-overhead direct GPU inference
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


# ============================================================
# LIVE CAMERA SETTINGS
# ============================================================

MODEL_PATH = r"rgb_finetuned.pt"

CAMERA_INDEX = 0          # Live camera device index (/dev/video0)
CAMERA_WIDTH = 1280       # High-speed capture resolution width
CAMERA_HEIGHT = 720       # High-speed capture resolution height
CAMERA_FPS = 60           # Targeted camera frame rate

CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.01
MAX_DISTANCE = 100
MAX_MISSING = 10

SHOW_DISPLAY = True       # Render GUI display window


# ============================================================
# HIGH-SPEED NATIVE TENSORRT ENGINE RUNNER (PURE GPU PIPELINE)
# ============================================================

class TRTInferenceEngine:
    """
    Direct TensorRT execution context bypasses PyTorch / Python overhead
    and runs pure CUDA memory inference on Jetson Orin Nano.
    """
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # Pre-allocate static GPU tensors for input/output to avoid per-frame allocations
        self.output_tensor = torch.zeros((1, 300, 6), dtype=torch.float32, device='cuda')
        self.canvas_gpu = torch.full((1, 3, 640, 640), 114.0 / 255.0, dtype=torch.float32, device='cuda')

        self.context.set_tensor_address('images', self.canvas_gpu.data_ptr())
        self.context.set_tensor_address('output0', self.output_tensor.data_ptr())

    def infer(self, frame_cpu, pad_offset_tensor, inv_r, pad_left, pad_top, new_w, new_h, conf_thresh):
        with torch.cuda.stream(self.stream):
            # Asynchronous GPU transfer
            fg_uint8 = torch.from_numpy(frame_cpu).to('cuda', non_blocking=True)
            # Permute BGR -> RGB & convert to float 0..1
            fg = fg_uint8[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
            fg.mul_(1.0 / 255.0)

            # Fast GPU bilinear letterbox resize
            resized = F.interpolate(fg, size=(new_h, new_w), mode='bilinear', align_corners=False)
            self.canvas_gpu[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

            # Execute TensorRT kernel on GPU stream
            self.context.execute_async_v3(self.stream.cuda_stream)

            # Post-process on GPU
            preds = self.output_tensor[0]  # (300, 6) -> [x1, y1, x2, y2, conf, class]
            mask = (preds[:, 5] == 0) & (preds[:, 4] >= conf_thresh)

            detections = []
            if mask.any():
                valid = preds[mask]
                scaled = (valid[:, :4] - pad_offset_tensor) * inv_r
                bboxes = scaled.cpu().numpy()
                confs = valid[:, 4].cpu().numpy()
                for i in range(len(bboxes)):
                    detections.append((
                        float(bboxes[i][0]),
                        float(bboxes[i][1]),
                        float(bboxes[i][2]),
                        float(bboxes[i][3]),
                        float(confs[i])
                    ))

        return detections


# ============================================================
# THREADED LIVE CAMERA CAPTURE (HARDWARE DECODED 60 FPS)
# ============================================================

class ThreadedVideoCapture:
    """
    Dedicated high-speed threaded capture for live USB / MIPI CSI cameras.
    Configures GStreamer / V4L2 MJPEG hardware decoding at 60 FPS.
    """
    def __init__(self, dev_index=0, width=1280, height=720, fps=60, queue_size=1):
        self.dev_index = dev_index
        self.cap = None

        # 1. Try GStreamer HW pipeline first (60 FPS MJPEG decode)
        gst_pipeline = (
            f"v4l2src device=/dev/video{dev_index} ! image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1"
        )
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        # 2. Fallback to V4L2 MJPEG if GStreamer pipeline fails
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(dev_index, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        # 3. Final fallback to standard OpenCV VideoCapture
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(dev_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            # Keep only the single latest live camera frame (zero latency)
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self):
        if self.stopped and self.q.empty():
            return False, None
        try:
            return self.q.get(timeout=0.2)
        except queue.Empty:
            return False, None

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap is not None:
            self.cap.release()


# ============================================================
# LOAD YOLO / TENSORRT MODEL
# ============================================================

print("Loading YOLO / TensorRT model...")

ENGINE_PATH = os.path.splitext(MODEL_PATH)[0] + ".engine"
trt_engine = None
model_fallback = None

if TRT_AVAILABLE and torch.cuda.is_available() and os.path.exists(ENGINE_PATH):
    try:
        print(f"Loading direct CUDA TensorRT FP16 Engine: {ENGINE_PATH}")
        trt_engine = TRTInferenceEngine(ENGINE_PATH)
        print("TensorRT Native CUDA Engine loaded successfully!")
    except Exception as e:
        print(f"Direct TensorRT engine load failed ({e}), falling back to Ultralytics wrapper...")
        trt_engine = None

if trt_engine is None:
    if os.path.exists(ENGINE_PATH):
        print(f"Loading TensorRT Engine via Ultralytics: {ENGINE_PATH}")
        model_fallback = YOLO(ENGINE_PATH, task="detect")
    else:
        print("TensorRT engine not found. Exporting FP16 TensorRT engine for Jetson Orin Nano...")
        try:
            model_pt = YOLO(MODEL_PATH)
            exported_path = model_pt.export(format="engine", half=True, device=0)
            model_fallback = YOLO(exported_path, task="detect")
            print(f"Successfully loaded exported TensorRT engine: {exported_path}")
        except Exception as e:
            print(f"TensorRT export failed ({e}). Falling back to PyTorch GPU FP16 mode...")
            model_fallback = YOLO(MODEL_PATH)

print("YOLO / TensorRT model ready.")


# ============================================================
# CREATE TRACKER
# ============================================================

tracker = DroneTracker(
    max_missing=MAX_MISSING,
    max_distance=MAX_DISTANCE,
    min_iou=IOU_THRESHOLD
)


# ============================================================
# OPEN LIVE CAMERA
# ============================================================

print(f"Opening live camera device index {CAMERA_INDEX} ({CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS)...")

cap = ThreadedVideoCapture(
    dev_index=CAMERA_INDEX,
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT,
    fps=CAMERA_FPS
)

if not cap.isOpened():
    print(f"ERROR: Cannot open camera device {CAMERA_INDEX}.")
    exit(1)


# ============================================================
# PRE-CALCULATE LETTERBOX PARAMETERS
# ============================================================

ret, sample_frame = cap.read()
if not ret or sample_frame is None:
    print("ERROR: Cannot read frame from camera.")
    exit(1)

h_orig, w_orig = sample_frame.shape[:2]
r = min(640.0 / h_orig, 640.0 / w_orig)
inv_r = 1.0 / r

new_w = int(round(w_orig * r))
new_h = int(round(h_orig * r))
dw = (640.0 - new_w) / 2.0
dh = (640.0 - new_h) / 2.0
top = int(round(dh - 0.1))
left = int(round(dw - 0.1))

# GPU letterbox offset tensor
if torch.cuda.is_available():
    pad_offset_tensor = torch.tensor([left, top, left, top], device='cuda', dtype=torch.float32)

# CPU canvas pre-allocation fallback
canvas = np.full((640, 640, 3), 114, dtype=np.uint8)


# Warmup run
if trt_engine is not None:
    trt_engine.infer(sample_frame, pad_offset_tensor, inv_r, left, top, new_w, new_h, CONFIDENCE_THRESHOLD)
else:
    cv2.resize(sample_frame, (new_w, new_h), dst=canvas[top:top+new_h, left:left+new_w], interpolation=cv2.INTER_LINEAR)
    model_fallback(canvas, verbose=False, conf=CONFIDENCE_THRESHOLD, device=0, half=True)


# ============================================================
# FPS & TRACKING STATE
# ============================================================

fps_start = time.perf_counter()
fps_counter = 0
fps = 0.0

tracking_started = False
previous_gray = None
frame_number = 0


# ============================================================
# MAIN LIVE CAMERA LOOP
# ============================================================

print("Live camera tracking loop started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame_number += 1

    # ========================================================
    # INFERENCE (DIRECT TENSORRT CUDA OR ULTRALYTICS FALLBACK)
    # ========================================================

    if trt_engine is not None:
        detections = trt_engine.infer(
            frame,
            pad_offset_tensor,
            inv_r,
            left,
            top,
            new_w,
            new_h,
            CONFIDENCE_THRESHOLD
        )
    else:
        cv2.resize(
            frame,
            (new_w, new_h),
            dst=canvas[top:top+new_h, left:left+new_w],
            interpolation=cv2.INTER_LINEAR
        )
        results = model_fallback(canvas, verbose=False, conf=CONFIDENCE_THRESHOLD, device=0, half=True)

        detections = []
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            mask = (clss == 0) & (confs >= CONFIDENCE_THRESHOLD)
            if np.any(mask):
                sub_xyxy = xyxy[mask]
                sub_confs = confs[mask]
                x1 = (sub_xyxy[:, 0] - left) * inv_r
                y1 = (sub_xyxy[:, 1] - top) * inv_r
                x2 = (sub_xyxy[:, 2] - left) * inv_r
                y2 = (sub_xyxy[:, 3] - top) * inv_r
                for i in range(len(sub_confs)):
                    detections.append((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(sub_confs[i])))


    # ========================================================
    # CONDITIONAL GRAYSCALE CONVERSION (OPTIMIZED FOR SPEED)
    # ========================================================

    gray = None
    if tracking_started and len(tracker.tracks) > 0:
        # Only perform grayscale conversion if optical flow is actually needed for lost tracks
        if any(track.missed > 0 for track in tracker.tracks):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # ========================================================
    # TRACKING UPDATE
    # ========================================================

    if not tracking_started:
        if len(detections) > 0:
            print(f"Drone detected on frame {frame_number}")
            tracker.reset()
            for detection in detections:
                tracker.create_track(detection[:4], detection[4])
            tracking_started = True
    else:
        tracker.update(detections, previous_gray, gray)

    if gray is not None:
        previous_gray = gray


    # ========================================================
    # RENDERING & DISPLAY
    # ========================================================

    if SHOW_DISPLAY:
        # Draw confirmed drone tracks
        if tracking_started:
            for track in tracker.tracks:
                if not track.confirmed:
                    continue

                bx1, by1, bx2, by2 = track.get_bbox()
                bx1 = max(0, min(w_orig - 1, bx1))
                by1 = max(0, min(h_orig - 1, by1))
                bx2 = max(0, min(w_orig - 1, bx2))
                by2 = max(0, min(h_orig - 1, by2))

                # Fast green bounding box
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2, cv2.LINE_8)

                # Track center red dot
                cx, cy = track.get_center()
                cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1, cv2.LINE_8)

                # Target label
                label = f"DRONE ID: {track.id} {track.confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (bx1, max(25, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_8
                )

        # Draw status overlay panel
        cv2.rectangle(frame, (10, 10), (330, 95), (20, 20, 20), -1)
        cv2.putText(frame, "YOLO Live Drone Tracker", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_8)
        status_str = "TRACKING" if tracking_started else "SEARCHING"
        cv2.putText(frame, f"Status: {status_str}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_8)
        cv2.putText(frame, f"Frame: {frame_number}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_8)

        # FPS counter update
        fps_counter += 1
        elapsed = time.perf_counter() - fps_start
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.perf_counter()

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (w_orig - 130, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_8
        )

        cv2.imshow("YOLO Live Drone Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
if SHOW_DISPLAY:
    cv2.destroyAllWindows()

print(f"Finished. Total camera frames processed: {frame_number}")