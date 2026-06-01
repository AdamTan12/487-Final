#include "app.h"

#include "camera.h"
#include "classifier.h"
#include "config.h"
#include "onnx_detector.h"
#include "preprocess.h"
#include "renderer.h"

#include <opencv2/highgui.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gr {

namespace {

constexpr const char* kWindowName = "gesture_recognition";
constexpr int         kEscKey     = 27;

std::vector<std::string> loadClassNames(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("App: failed to open class_names: " + path);
    }
    std::vector<std::string> names;
    for (std::string line; std::getline(in, line); ) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) names.push_back(line);
    }
    if (names.empty()) {
        throw std::runtime_error("App: empty class_names: " + path);
    }
    return names;
}

}  // namespace

App::App(AppConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.debug) {
        std::cerr << "App: camera=" << cfg_.camera_index
                  << " model=" << cfg_.model_path
                  << " detector=" << cfg_.detector_model_path << "\n";
    }
}

App::~App() = default;

void App::runLive() {
    Camera camera(cfg_.camera_index);
    if (!camera.isOpen()) {
        throw std::runtime_error("App::runLive: camera failed to open");
    }

    OnnxDetector detector(cfg_.detector_model_path);
    Classifier   classifier(cfg_.model_path);
    Renderer     renderer(loadClassNames(cfg_.class_names_path));

    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);

    using clock = std::chrono::steady_clock;
    auto   last_fps_t  = clock::now();
    int    fps_frames  = 0;
    double fps_display = 0.0;

    cv::Mat frame;
    while (true) {
        if (!camera.readFrame(frame)) {
            std::cerr << "App::runLive: dropped frame, retrying\n";
            continue;
        }

        std::optional<cv::Rect> bbox = detector.detect(frame);

        int   class_id   = -1;
        float confidence = 0.0f;
        std::vector<float> scores;
        if (bbox) {
            cv::Mat blob   = prepareBlob(frame, *bbox, cfg_);
            scores         = classifier.classify(blob);
            const auto it  = std::max_element(scores.begin(), scores.end());
            class_id       = static_cast<int>(std::distance(scores.begin(), it));
            confidence     = *it;
        }

        // Rolling FPS, refreshed twice a second so the number isn't jittery.
        ++fps_frames;
        const auto now = clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_fps_t).count();
        if (elapsed >= 0.5) {
            fps_display = fps_frames / elapsed;
            fps_frames  = 0;
            last_fps_t  = now;
        }

        renderer.draw(frame, bbox, class_id, confidence, scores, fps_display);
        cv::imshow(kWindowName, frame);
        if (cv::waitKey(1) == kEscKey) {
            break;
        }
    }

    cv::destroyWindow(kWindowName);
}

}  // namespace gr
