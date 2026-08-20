import cv2
import numpy as np
import math


class DroneTrack:

    def __init__(self, bbox, track_id):

        self.id = track_id

        x1, y1, x2, y2 = bbox

        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

        self.width = max(2.0, self.x2 - self.x1)
        self.height = max(2.0, self.y2 - self.y1)

        self.cx = (self.x1 + self.x2) / 2
        self.cy = (self.y1 + self.y2) / 2

        # ====================================================
        # KALMAN FILTER
        #
        # State:
        #
        # x
        # y
        # vx
        # vy
        # width
        # height
        # ====================================================

        self.kalman = cv2.KalmanFilter(6, 4)

        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ],
            dtype=np.float32
        )

        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ],
            dtype=np.float32
        )

        # Process noise
        self.kalman.processNoiseCov = np.eye(
            6,
            dtype=np.float32
        ) * 0.03

        # Measurement noise
        self.kalman.measurementNoiseCov = np.eye(
            4,
            dtype=np.float32
        ) * 0.15

        self.kalman.errorCovPost = np.eye(
            6,
            dtype=np.float32
        )

        self.kalman.statePost = np.array(
            [
                [self.cx],
                [self.cy],
                [0],
                [0],
                [self.width],
                [self.height]
            ],
            dtype=np.float32
        )

        # ====================================================
        # TRACK STATE
        # ====================================================

        self.age = 1

        self.hits = 1

        self.missed = 0

        self.confirmed = False

        self.confidence = 0.0

        self.history = []

        self.last_gray = None

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self):

        prediction = self.kalman.predict()

        self.cx = float(prediction[0][0])
        self.cy = float(prediction[1][0])

        vx = float(prediction[2][0])
        vy = float(prediction[3][0])

        self.width = max(
            2.0,
            float(prediction[4][0])
        )

        self.height = max(
            2.0,
            float(prediction[5][0])
        )

        self.x1 = self.cx - self.width / 2
        self.y1 = self.cy - self.height / 2

        self.x2 = self.cx + self.width / 2
        self.y2 = self.cy + self.height / 2

        self.age += 1

        self.missed += 1

        return self.get_bbox()

    # ========================================================
    # UPDATE FROM YOLO
    # ========================================================

    def update(self, bbox, confidence):

        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        width = max(
            2.0,
            x2 - x1
        )

        height = max(
            2.0,
            y2 - y1
        )

        measurement = np.array(
            [
                [cx],
                [cy],
                [width],
                [height]
            ],
            dtype=np.float32
        )

        self.kalman.correct(
            measurement
        )

        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

        self.width = width
        self.height = height

        self.cx = cx
        self.cy = cy

        self.confidence = confidence

        self.hits += 1

        self.missed = 0

        self.history.append(
            (cx, cy)
        )

        if len(self.history) > 30:
            self.history.pop(0)

        # Confirm after several successful updates

        if self.hits >= 3:
            self.confirmed = True

    # ========================================================
    # OPTICAL FLOW REFINEMENT
    # ========================================================

    def optical_flow_update(
        self,
        old_gray,
        new_gray
    ):

        if old_gray is None:
            return False

        x1, y1, x2, y2 = self.get_bbox()

        h, w = new_gray.shape

        # Expand region slightly

        padding = 5

        x1 = max(
            0,
            x1 - padding
        )

        y1 = max(
            0,
            y1 - padding
        )

        x2 = min(
            w - 1,
            x2 + padding
        )

        y2 = min(
            h - 1,
            y2 + padding
        )

        if x2 <= x1 or y2 <= y1:
            return False

        # ----------------------------------------------------
        # Create feature points around drone
        # ----------------------------------------------------

        roi = old_gray[
            y1:y2,
            x1:x2
        ]

        if roi.size == 0:
            return False

        points = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=20,
            qualityLevel=0.01,
            minDistance=2,
            blockSize=3
        )

        if points is None:
            return False

        points[:, :, 0] += x1
        points[:, :, 1] += y1

        # ----------------------------------------------------
        # Optical flow
        # ----------------------------------------------------

        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray,
            new_gray,
            points,
            None,
            winSize=(15, 15),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                20,
                0.03
            )
        )

        if new_points is None:
            return False

        good_old = points[
            status == 1
        ]

        good_new = new_points[
            status == 1
        ]

        if len(good_new) < 3:
            return False

        # ----------------------------------------------------
        # Calculate movement
        # ----------------------------------------------------

        movement = good_new - good_old

        dx = np.median(
            movement[:, 0]
        )

        dy = np.median(
            movement[:, 1]
        )

        # ----------------------------------------------------
        # Update center
        # ----------------------------------------------------

        self.cx += dx
        self.cy += dy

        self.x1 = (
            self.cx -
            self.width / 2
        )

        self.y1 = (
            self.cy -
            self.height / 2
        )

        self.x2 = (
            self.cx +
            self.width / 2
        )

        self.y2 = (
            self.cy +
            self.height / 2
        )

        return True

    # ========================================================
    # BOUNDING BOX
    # ========================================================

    def get_bbox(self):

        return (
            int(self.x1),
            int(self.y1),
            int(self.x2),
            int(self.y2)
        )

    # ========================================================
    # CENTER
    # ========================================================

    def get_center(self):

        return (
            float(self.cx),
            float(self.cy)
        )

    # ========================================================
    # VELOCITY
    # ========================================================

    def get_velocity(self):

        state = self.kalman.statePost

        return (
            float(state[2][0]),
            float(state[3][0])
        )


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
    )

    union = (
        area1 +
        area2 -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# CENTER DISTANCE
# ============================================================

def center_distance(box1, box2):

    c1x = (
        box1[0] +
        box1[2]
    ) / 2

    c1y = (
        box1[1] +
        box1[3]
    ) / 2

    c2x = (
        box2[0] +
        box2[2]
    ) / 2

    c2y = (
        box2[1] +
        box2[3]
    ) / 2

    return math.sqrt(
        (c1x - c2x) ** 2 +
        (c1y - c2y) ** 2
    )


# ============================================================
# DRONE TRACKER
# ============================================================

class DroneTracker:

    def __init__(
        self,

        max_missing=10,

        max_distance=100,

        min_iou=0.01
    ):

        self.tracks = []

        self.next_id = 1

        self.max_missing = max_missing

        self.max_distance = max_distance

        self.min_iou = min_iou

    # ========================================================
    # START TRACKING
    # ========================================================

    def create_track(
        self,
        bbox,
        confidence
    ):

        track = DroneTrack(
            bbox,
            self.next_id
        )

        track.confidence = confidence

        self.next_id += 1

        self.tracks.append(
            track
        )

        return track

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        detections,
        old_gray=None,
        new_gray=None
    ):

        # ====================================================
        # STEP 1
        # Predict existing tracks
        # ====================================================

        predicted_boxes = []

        for track in self.tracks:

            predicted_boxes.append(
                track.predict()
            )

        # ====================================================
        # STEP 2
        # Build possible matches
        # ====================================================

        matches = []

        for track_index, track in enumerate(
            self.tracks
        ):

            predicted_box = predicted_boxes[
                track_index
            ]

            for detection_index, detection in enumerate(
                detections
            ):

                detection_box = detection[
                    :4
                ]

                iou = calculate_iou(
                    predicted_box,
                    detection_box
                )

                distance = center_distance(
                    predicted_box,
                    detection_box
                )

                # --------------------------------------------
                # Adaptive distance
                # --------------------------------------------

                box_width = (
                    predicted_box[2] -
                    predicted_box[0]
                )

                box_height = (
                    predicted_box[3] -
                    predicted_box[1]
                )

                adaptive_distance = max(
                    self.max_distance,
                    max(
                        box_width,
                        box_height
                    ) * 3
                )

                # --------------------------------------------
                # Reject impossible matches
                # --------------------------------------------

                if (
                    iou < self.min_iou
                    and
                    distance > adaptive_distance
                ):
                    continue

                # --------------------------------------------
                # Normalize distance
                # --------------------------------------------

                distance_score = min(
                    1.0,
                    distance /
                    adaptive_distance
                )

                # --------------------------------------------
                # Cost
                #
                # Lower is better
                # --------------------------------------------

                cost = (

                    0.60 *
                    (1.0 - iou)

                    +

                    0.40 *
                    distance_score

                )

                matches.append(
                    (
                        cost,
                        track_index,
                        detection_index
                    )
                )

        # ====================================================
        # STEP 3
        # Sort matches
        # ====================================================

        matches.sort(
            key=lambda x: x[0]
        )

        used_tracks = set()

        used_detections = set()

        final_matches = []

        for (
            cost,
            track_index,
            detection_index
        ) in matches:

            if track_index in used_tracks:
                continue

            if detection_index in used_detections:
                continue

            final_matches.append(
                (
                    track_index,
                    detection_index
                )
            )

            used_tracks.add(
                track_index
            )

            used_detections.add(
                detection_index
            )

        # ====================================================
        # STEP 4
        # Update matched tracks
        # ====================================================

        for (
            track_index,
            detection_index
        ) in final_matches:

            detection = detections[
                detection_index
            ]

            bbox = detection[
                :4
            ]

            confidence = detection[
                4
            ]

            self.tracks[
                track_index
            ].update(
                bbox,
                confidence
            )

        # ====================================================
        # STEP 5
        # Optical flow for tracks that were not detected
        # ====================================================

        for index, track in enumerate(
            self.tracks
        ):

            if index in used_tracks:
                continue

            if (
                old_gray is not None
                and
                new_gray is not None
            ):

                track.optical_flow_update(
                    old_gray,
                    new_gray
                )

        # ====================================================
        # STEP 6
        # Create new tracks from unmatched detections
        # ====================================================

        for detection_index, detection in enumerate(
            detections
        ):

            if detection_index in used_detections:
                continue

            self.create_track(
                detection[:4],
                detection[4]
            )

        # ====================================================
        # STEP 7
        # Delete lost tracks
        # ====================================================

        remaining_tracks = []

        for track in self.tracks:

            if track.missed <= self.max_missing:

                remaining_tracks.append(
                    track
                )

        self.tracks = remaining_tracks

        # ====================================================
        # Return active tracks
        # ====================================================

        return [
            track

            for track in self.tracks

            if track.confirmed
        ]

    # ========================================================
    # CLEAR
    # ========================================================

    def reset(self):

        self.tracks = []

        self.next_id = 1