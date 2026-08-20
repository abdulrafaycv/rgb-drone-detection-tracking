#include "drone_tracker.hpp"

// ============================================================
// IOU CALCULATION
// ============================================================

float calculate_iou(const cv::Vec4f& box1, const cv::Vec4f& box2) {
    float x1 = std::max(box1[0], box2[0]);
    float y1 = std::max(box1[1], box2[1]);
    float x2 = std::min(box1[2], box2[2]);
    float y2 = std::min(box1[3], box2[3]);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = std::max(0.0f, box1[2] - box1[0]) * std::max(0.0f, box1[3] - box1[1]);
    float area2 = std::max(0.0f, box2[2] - box2[0]) * std::max(0.0f, box2[3] - box2[1]);
    float union_area = area1 + area2 - intersection;

    if (union_area <= 0.0f) return 0.0f;
    return intersection / union_area;
}

float calculate_iou(const cv::Vec4i& box1, const cv::Vec4f& box2) {
    cv::Vec4f b1((float)box1[0], (float)box1[1], (float)box1[2], (float)box1[3]);
    return calculate_iou(b1, box2);
}

// ============================================================
// CENTER DISTANCE
// ============================================================

float center_distance(const cv::Vec4i& box1, const cv::Vec4f& box2) {
    float c1x = (box1[0] + box1[2]) / 2.0f;
    float c1y = (box1[1] + box1[3]) / 2.0f;
    float c2x = (box2[0] + box2[2]) / 2.0f;
    float c2y = (box2[1] + box2[3]) / 2.0f;

    float dx = c1x - c2x;
    float dy = c1y - c2y;
    return std::sqrt(dx * dx + dy * dy);
}

// ============================================================
// DRONE TRACK CONSTRUCTOR
// ============================================================

DroneTrack::DroneTrack(const cv::Vec4f& bbox, int track_id)
    : id(track_id),
      x1(bbox[0]), y1(bbox[1]), x2(bbox[2]), y2(bbox[3]),
      age(1), hits(1), missed(0), confirmed(false), confidence(0.0f)
{
    width = std::max(2.0f, x2 - x1);
    height = std::max(2.0f, y2 - y1);
    cx = (x1 + x2) / 2.0f;
    cy = (y1 + y2) / 2.0f;

    kalman.init(6, 4, 0, CV_32F);

    // Transition matrix
    cv::setIdentity(kalman.transitionMatrix);
    kalman.transitionMatrix.at<float>(0, 2) = 1.0f;
    kalman.transitionMatrix.at<float>(1, 3) = 1.0f;

    // Measurement matrix
    kalman.measurementMatrix = cv::Mat::zeros(4, 6, CV_32F);
    kalman.measurementMatrix.at<float>(0, 0) = 1.0f;
    kalman.measurementMatrix.at<float>(1, 1) = 1.0f;
    kalman.measurementMatrix.at<float>(2, 4) = 1.0f;
    kalman.measurementMatrix.at<float>(3, 5) = 1.0f;

    // Process noise
    cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(0.03f));

    // Measurement noise
    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(0.15f));

    // Error covariance post
    cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1.0f));

    // State post
    kalman.statePost.at<float>(0) = cx;
    kalman.statePost.at<float>(1) = cy;
    kalman.statePost.at<float>(2) = 0.0f;
    kalman.statePost.at<float>(3) = 0.0f;
    kalman.statePost.at<float>(4) = width;
    kalman.statePost.at<float>(5) = height;
}

// ============================================================
// KALMAN PREDICT
// ============================================================

cv::Vec4i DroneTrack::predict() {
    cv::Mat prediction = kalman.predict();

    cx = prediction.at<float>(0);
    cy = prediction.at<float>(1);
    width = std::max(2.0f, prediction.at<float>(4));
    height = std::max(2.0f, prediction.at<float>(5));

    x1 = cx - width / 2.0f;
    y1 = cy - height / 2.0f;
    x2 = cx + width / 2.0f;
    y2 = cy + height / 2.0f;

    age++;
    missed++;

    return get_bbox();
}

// ============================================================
// KALMAN UPDATE FROM YOLO
// ============================================================

void DroneTrack::update(const cv::Vec4f& bbox, float conf) {
    float n_x1 = bbox[0];
    float n_y1 = bbox[1];
    float n_x2 = bbox[2];
    float n_y2 = bbox[3];

    float n_cx = (n_x1 + n_x2) / 2.0f;
    float n_cy = (n_y1 + n_y2) / 2.0f;
    float n_w = std::max(2.0f, n_x2 - n_x1);
    float n_h = std::max(2.0f, n_y2 - n_y1);

    cv::Mat measurement(4, 1, CV_32F);
    measurement.at<float>(0) = n_cx;
    measurement.at<float>(1) = n_cy;
    measurement.at<float>(2) = n_w;
    measurement.at<float>(3) = n_h;

    kalman.correct(measurement);

    x1 = n_x1;
    y1 = n_y1;
    x2 = n_x2;
    y2 = n_y2;
    width = n_w;
    height = n_h;
    cx = n_cx;
    cy = n_cy;

    confidence = conf;
    hits++;
    missed = 0;

    if (hits >= 3) {
        confirmed = true;
    }
}

// ============================================================
// OPTICAL FLOW REFINEMENT
// ============================================================

static float compute_median(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    if (v.size() % 2 != 0) return v[n];
    auto max_it = std::max_element(v.begin(), v.begin() + n);
    return (*max_it + v[n]) / 2.0f;
}

bool DroneTrack::optical_flow_update(const cv::Mat& old_gray, const cv::Mat& new_gray) {
    if (old_gray.empty() || new_gray.empty()) return false;

    cv::Vec4i bbox = get_bbox();
    int h = new_gray.rows;
    int w = new_gray.cols;

    int padding = 5;
    int rx1 = std::max(0, bbox[0] - padding);
    int ry1 = std::max(0, bbox[1] - padding);
    int rx2 = std::min(w - 1, bbox[2] + padding);
    int ry2 = std::min(h - 1, bbox[3] + padding);

    if (rx2 <= rx1 || ry2 <= ry1) return false;

    cv::Mat roi = old_gray(cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));
    if (roi.empty()) return false;

    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(roi, points, 20, 0.01, 2, cv::Mat(), 3);
    if (points.empty()) return false;

    for (auto& pt : points) {
        pt.x += rx1;
        pt.y += ry1;
    }

    std::vector<cv::Point2f> new_points;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(
        old_gray, new_gray, points, new_points, status, err,
        cv::Size(15, 15), 3,
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.03)
    );

    std::vector<float> dx_list;
    std::vector<float> dy_list;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            dx_list.push_back(new_points[i].x - points[i].x);
            dy_list.push_back(new_points[i].y - points[i].y);
        }
    }

    if (dx_list.size() < 3) return false;

    float dx = compute_median(dx_list);
    float dy = compute_median(dy_list);

    cx += dx;
    cy += dy;
    x1 = cx - width / 2.0f;
    y1 = cy - height / 2.0f;
    x2 = cx + width / 2.0f;
    y2 = cy + height / 2.0f;

    return true;
}

cv::Vec4i DroneTrack::get_bbox() const {
    return cv::Vec4i((int)x1, (int)y1, (int)x2, (int)y2);
}

cv::Point2f DroneTrack::get_center() const {
    return cv::Point2f(cx, cy);
}

// ============================================================
// DRONE TRACKER CONSTRUCTOR
// ============================================================

DroneTracker::DroneTracker(int max_missing, float max_distance, float min_iou)
    : next_id(1), max_missing(max_missing), max_distance(max_distance), min_iou(min_iou) {}

void DroneTracker::reset() {
    tracks.clear();
    next_id = 1;
}

DroneTrack& DroneTracker::create_track(const cv::Vec4f& bbox, float confidence) {
    tracks.emplace_back(bbox, next_id++);
    tracks.back().confidence = confidence;
    return tracks.back();
}

// Struct for sorting cost matches
struct MatchCost {
    float cost;
    int track_idx;
    int det_idx;
};

std::vector<DroneTrack*> DroneTracker::update(const std::vector<Detection>& detections,
                                              const cv::Mat& old_gray,
                                              const cv::Mat& new_gray)
{
    // Step 1: Predict existing tracks
    std::vector<cv::Vec4i> predicted_boxes;
    for (auto& track : tracks) {
        predicted_boxes.push_back(track.predict());
    }

    // Step 2: Build possible matches
    std::vector<MatchCost> matches;

    for (size_t t_idx = 0; t_idx < tracks.size(); ++t_idx) {
        const cv::Vec4i& p_box = predicted_boxes[t_idx];
        float p_w = (float)(p_box[2] - p_box[0]);
        float p_h = (float)(p_box[3] - p_box[1]);

        float adaptive_distance = std::max(max_distance, std::max(p_w, p_h) * 3.0f);

        for (size_t d_idx = 0; d_idx < detections.size(); ++d_idx) {
            cv::Vec4f d_box(detections[d_idx].x1, detections[d_idx].y1, detections[d_idx].x2, detections[d_idx].y2);

            float iou = calculate_iou(p_box, d_box);
            float dist = center_distance(p_box, d_box);

            if (iou < min_iou && dist > adaptive_distance) {
                continue;
            }

            float distance_score = std::min(1.0f, dist / adaptive_distance);
            float cost = 0.60f * (1.0f - iou) + 0.40f * distance_score;

            matches.push_back({cost, (int)t_idx, (int)d_idx});
        }
    }

    // Step 3: Sort matches by ascending cost
    std::sort(matches.begin(), matches.end(), [](const MatchCost& a, const MatchCost& b) {
        return a.cost < b.cost;
    });

    std::vector<bool> used_tracks(tracks.size(), false);
    std::vector<bool> used_detections(detections.size(), false);
    std::vector<std::pair<int, int>> final_matches;

    for (const auto& m : matches) {
        if (used_tracks[m.track_idx] || used_detections[m.det_idx]) {
            continue;
        }
        final_matches.push_back({m.track_idx, m.det_idx});
        used_tracks[m.track_idx] = true;
        used_detections[m.det_idx] = true;
    }

    // Step 4: Update matched tracks
    for (const auto& fm : final_matches) {
        const auto& det = detections[fm.second];
        cv::Vec4f bbox(det.x1, det.y1, det.x2, det.y2);
        tracks[fm.first].update(bbox, det.confidence);
    }

    // Step 5: Optical flow for unmatched tracks
    if (!old_gray.empty() && !new_gray.empty()) {
        for (size_t i = 0; i < tracks.size(); ++i) {
            if (!used_tracks[i]) {
                tracks[i].optical_flow_update(old_gray, new_gray);
            }
        }
    }

    // Step 6: Create new tracks from unmatched detections
    for (size_t d_idx = 0; d_idx < detections.size(); ++d_idx) {
        if (!used_detections[d_idx]) {
            const auto& det = detections[d_idx];
            cv::Vec4f bbox(det.x1, det.y1, det.x2, det.y2);
            create_track(bbox, det.confidence);
        }
    }

    // Step 7: Delete lost tracks
    std::vector<DroneTrack> remaining_tracks;
    for (const auto& track : tracks) {
        if (track.missed <= max_missing) {
            remaining_tracks.push_back(track);
        }
    }
    tracks = std::move(remaining_tracks);

    // Return confirmed active tracks
    std::vector<DroneTrack*> active_tracks;
    for (auto& track : tracks) {
        if (track.confirmed) {
            active_tracks.push_back(&track);
        }
    }

    return active_tracks;
}
