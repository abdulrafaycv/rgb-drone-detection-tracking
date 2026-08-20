#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <memory>

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "drone_tracker.hpp"

// ============================================================
// SETTINGS (ACCURACY & PIPELINE PARAMETERS)
// ============================================================

const std::string MODEL_PATH = "rgb_finetuned.engine";
const int CAMERA_INDEX = 0; // Live camera index (0 = /dev/video0)

const float CONFIDENCE_THRESHOLD = 0.60f;
const float IOU_THRESHOLD = 0.01f;
const float MAX_DISTANCE = 100.0f;
const int MAX_MISSING = 10;
const int DRONE_CLASS_ID = 0;

// ============================================================
// TENSORRT LOGGER
// ============================================================

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// ============================================================
// THREADED VIDEO CAPTURE (C++)
// ============================================================

class ThreadedVideoCapture {
private:
    cv::VideoCapture cap;
    std::queue<cv::Mat> q;
    std::mutex mtx;
    std::condition_variable cv_q;
    std::thread reader_thread;
    bool stopped;
    size_t max_queue_size;

    void reader_loop() {
        int empty_count = 0;
        while (!stopped) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                empty_count++;
                if (empty_count > 30) {
                    std::unique_lock<std::mutex> lock(mtx);
                    stopped = true;
                    cv_q.notify_all();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            empty_count = 0;

            std::unique_lock<std::mutex> lock(mtx);
            // Drop oldest frame if queue is full to eliminate latency and maintain real-time throughput
            while (q.size() >= max_queue_size) {
                q.pop();
            }
            q.push(frame.clone());
            cv_q.notify_all();
        }
    }

public:
    ThreadedVideoCapture(int device_id = 0, size_t queue_size = 2)
        : stopped(false), max_queue_size(queue_size)
    {
        // Try opening with V4L2 backend to configure high-speed camera properties
        cap.open(device_id, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            cap.open(device_id); // Fallback to default backend
        }

        if (cap.isOpened()) {
            // MJPEG format prevents USB bandwidth bottlenecks (uncompressed YUYV defaults to 5 FPS)
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            cap.set(cv::CAP_PROP_FPS, 60);

            reader_thread = std::thread(&ThreadedVideoCapture::reader_loop, this);
        }
    }

    ThreadedVideoCapture(const std::string& src, size_t queue_size = 2)
        : cap(src), stopped(false), max_queue_size(queue_size)
    {
        if (cap.isOpened()) {
            reader_thread = std::thread(&ThreadedVideoCapture::reader_loop, this);
        }
    }

    ~ThreadedVideoCapture() {
        release();
    }

    bool isOpened() const {
        return cap.isOpened();
    }

    bool read(cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_q.wait(lock, [this] { return !q.empty() || stopped; });

        if (q.empty()) {
            return false;
        }

        frame = q.front();
        q.pop();
        cv_q.notify_all();
        return true;
    }

    void release() {
        stopped = true;
        cv_q.notify_all();
        if (reader_thread.joinable()) {
            reader_thread.join();
        }
        cap.release();
    }
};

// ============================================================
// MAIN FUNCTION
// ============================================================

int main() {
    std::cout << "Loading TensorRT model engine: " << MODEL_PATH << std::endl;

    std::ifstream file(MODEL_PATH, std::ios::binary);
    if (!file.good()) {
        std::cerr << "ERROR: Cannot open TensorRT engine file: " << MODEL_PATH << std::endl;
        return -1;
    }

    file.seekg(0, file.end);
    size_t engine_size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engine_data(engine_size);
    file.read(engine_data.data(), engine_size);
    file.close();

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engine_data.data(), engine_size));

    if (!engine) {
        std::cerr << "ERROR: Failed to deserialize CUDA engine." << std::endl;
        return -1;
    }

    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    std::cout << "TensorRT Engine successfully loaded." << std::endl;

    // Allocate GPU and Host buffers for TensorRT
    const int INPUT_H = 640;
    const int INPUT_W = 640;
    const int INPUT_C = 3;
    const int OUTPUT_BOXES = 300;
    const int OUTPUT_COLS = 6;

    size_t input_bytes = 1 * INPUT_C * INPUT_H * INPUT_W * sizeof(float);
    size_t output_bytes = 1 * OUTPUT_BOXES * OUTPUT_COLS * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc((void**)&d_input, input_bytes);
    cudaMalloc((void**)&d_output, output_bytes);

    std::vector<float> h_input(1 * INPUT_C * INPUT_H * INPUT_W);
    std::vector<float> h_output(1 * OUTPUT_BOXES * OUTPUT_COLS);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    context->setTensorAddress("images", d_input);
    context->setTensorAddress("output0", d_output);

    // Create Tracker
    DroneTracker tracker(MAX_MISSING, MAX_DISTANCE, IOU_THRESHOLD);

    // Live Camera Reader
    ThreadedVideoCapture cap(CAMERA_INDEX);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open live camera device at index: " << CAMERA_INDEX << std::endl;
        return -1;
    }

    cv::Mat sample_frame;
    if (!cap.read(sample_frame) || sample_frame.empty()) {
        std::cerr << "ERROR: Cannot read sample frame." << std::endl;
        return -1;
    }

    // Pre-calculate fast canvas letterbox math
    int h_orig = sample_frame.rows;
    int w_orig = sample_frame.cols;

    float r = std::min(640.0f / (float)h_orig, 640.0f / (float)w_orig);
    int new_w = (int)std::round(w_orig * r);
    int new_h = (int)std::round(h_orig * r);

    float dw = (640.0f - (float)new_w) / 2.0f;
    float dh = (640.0f - (float)new_h) / 2.0f;
    int top = (int)std::round(dh - 0.1f);
    int left = (int)std::round(dw - 0.1f);

    // Pre-allocated static canvas
    cv::Mat canvas(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat canvas_roi = canvas(cv::Rect(left, top, new_w, new_h));

    // Pipeline timing & tracking state
    auto total_start = std::chrono::high_resolution_clock::now();
    auto fps_start = std::chrono::high_resolution_clock::now();
    int fps_counter = 0;
    float fps = 0.0f;

    bool tracking_started = false;
    cv::Mat previous_gray;
    cv::Mat current_gray;
    int frame_number = 0;

    // ============================================================
    // MAIN LOOP
    // ============================================================

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            break;
        }

        frame_number++;

        // 1. Fast GPU pre-letterbox resize into static canvas
        cv::resize(frame, canvas_roi, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

        // 2. Fast SIMD Preprocess: HWC BGR uint8 -> CHW RGB float32 [0, 1]
        cv::Mat blob;
        cv::dnn::blobFromImage(canvas, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false, CV_32F);

        // Copy host input to device
        cudaMemcpyAsync(d_input, blob.ptr<float>(), input_bytes, cudaMemcpyHostToDevice, stream);

        // 3. TensorRT Execution
        context->enqueueV3(stream);

        // Copy device output to host
        cudaMemcpyAsync(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // 4. Parse Detections
        std::vector<Detection> detections;

        for (int i = 0; i < OUTPUT_BOXES; ++i) {
            float x1_640 = h_output[i * 6 + 0];
            float y1_640 = h_output[i * 6 + 1];
            float x2_640 = h_output[i * 6 + 2];
            float y2_640 = h_output[i * 6 + 3];
            float confidence = h_output[i * 6 + 4];
            int cls = (int)h_output[i * 6 + 5];

            if (cls != DRONE_CLASS_ID || confidence < CONFIDENCE_THRESHOLD) {
                continue;
            }

            // Un-pad bboxes back to original image space
            float x1 = (x1_640 - (float)left) / r;
            float y1 = (y1_640 - (float)top) / r;
            float x2 = (x2_640 - (float)left) / r;
            float y2 = (y2_640 - (float)top) / r;

            detections.push_back({x1, y1, x2, y2, confidence});
        }

        // 5. Optional Grayscale Conversion (Only when required by optical flow)
        current_gray.release();
        if (tracking_started && !tracker.tracks.empty()) {
            cv::cvtColor(frame, current_gray, cv::COLOR_BGR2GRAY);
        }

        // 6. Drone Tracker Update
        if (!tracking_started) {
            if (!detections.empty()) {
                std::cout << "Drone detected on frame " << frame_number << std::endl;
                tracker.reset();
                for (const auto& det : detections) {
                    cv::Vec4f bbox(det.x1, det.y1, det.x2, det.y2);
                    tracker.create_track(bbox, det.confidence);
                }
                tracking_started = true;
            }
        } else {
            tracker.update(detections, previous_gray, current_gray);
        }

        if (!current_gray.empty()) {
            previous_gray = current_gray;
        }

        // 7. Draw Visual HUD & Tracks
        if (tracking_started) {
            for (const auto& track : tracker.tracks) {
                if (!track.confirmed) continue;

                cv::Vec4i bbox = track.get_bbox();
                int x1 = std::max(0, std::min(frame.cols - 1, bbox[0]));
                int y1 = std::max(0, std::min(frame.rows - 1, bbox[1]));
                int x2 = std::max(0, std::min(frame.cols - 1, bbox[2]));
                int y2 = std::max(0, std::min(frame.rows - 1, bbox[3]));

                // Green Box
                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

                // Red Center Dot
                cv::Point2f center = track.get_center();
                cv::circle(frame, cv::Point((int)center.x, (int)center.y), 3, cv::Scalar(0, 0, 255), -1);

                // Text Label
                char label_buf[128];
                snprintf(label_buf, sizeof(label_buf), "DRONE ID: %d %.2f", track.id, track.confidence);
                cv::putText(frame, label_buf, cv::Point(x1, std::max(25, y1 - 8)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }
        }

        // Status Panel Overlay
        cv::rectangle(frame, cv::Point(10, 10), cv::Point(330, 95), cv::Scalar(20, 20, 20), -1);
        cv::putText(frame, "YOLO26n + OpenCV Tracker (C++)", cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        std::string status_str = tracking_started ? "Status: TRACKING" : "Status: SEARCHING";
        cv::putText(frame, status_str, cv::Point(20, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);

        char frame_buf[64];
        snprintf(frame_buf, sizeof(frame_buf), "Frame: %d", frame_number);
        cv::putText(frame, frame_buf, cv::Point(20, 82),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);

        // Calculate FPS
        fps_counter++;
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(now - fps_start).count();
        if (elapsed >= 1.0f) {
            fps = (float)fps_counter / elapsed;
            fps_counter = 0;
            fps_start = now;
        }

        char fps_buf[64];
        snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);
        cv::putText(frame, fps_buf, cv::Point(frame.cols - 130, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2);

        if (frame_number % 50 == 0) {
            std::cout << "Frame " << frame_number << " | Current FPS: " << fps << std::endl;
        }

        // Display Window (Open GUI only if DISPLAY env var is set)
        const char* display_env = std::getenv("DISPLAY");
        if (display_env != nullptr && std::string(display_env).length() > 0) {
            cv::imshow("YOLO26n + OpenCV Drone Tracker (C++)", frame);
            char key = (char)cv::waitKey(1);
            if (key == 'q' || key == 'Q') {
                break;
            }
        }
    }

    // Cleanup GPU Memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    cap.release();
    cv::destroyAllWindows();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_duration = std::chrono::duration<float>(total_end - total_start).count();

    std::cout << "Finished C++ Tracker execution." << std::endl;
    std::cout << "Processed " << frame_number << " frames in " << total_duration << " s -> Average C++ Pipeline FPS: " << (frame_number / total_duration) << std::endl;
    return 0;
}
