#ifndef DRONE_TRACKER_HPP
#define DRONE_TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
};

class DroneTrack {
public:
    int id;
    float x1, y1, x2, y2;
    float width, height;
    float cx, cy;

    cv::KalmanFilter kalman;

    int age;
    int hits;
    int missed;
    bool confirmed;
    float confidence;

    DroneTrack(const cv::Vec4f& bbox, int track_id);

    cv::Vec4i predict();
    void update(const cv::Vec4f& bbox, float conf);
    bool optical_flow_update(const cv::Mat& old_gray, const cv::Mat& new_gray);

    cv::Vec4i get_bbox() const;
    cv::Point2f get_center() const;
};

class DroneTracker {
public:
    std::vector<DroneTrack> tracks;
    int next_id;
    int max_missing;
    float max_distance;
    float min_iou;

    DroneTracker(int max_missing = 10, float max_distance = 100.0f, float min_iou = 0.01f);

    void reset();
    DroneTrack& create_track(const cv::Vec4f& bbox, float confidence);
    std::vector<DroneTrack*> update(const std::vector<Detection>& detections,
                                   const cv::Mat& old_gray = cv::Mat(),
                                   const cv::Mat& new_gray = cv::Mat());
};

float calculate_iou(const cv::Vec4f& box1, const cv::Vec4f& box2);
float calculate_iou(const cv::Vec4i& box1, const cv::Vec4f& box2);
float center_distance(const cv::Vec4i& box1, const cv::Vec4f& box2);

#endif // DRONE_TRACKER_HPP
