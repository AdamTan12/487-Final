#include "app.h"

#include "camera.h"
#include "config.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace gr {

namespace {

constexpr const char* kWindowName = "gesture_recognition";
constexpr int kEscKey = 27;

std::string modeLabel(AppMode mode) {
    switch (mode) {
        case AppMode::Live:  return "live";
        case AppMode::Eval:  return "eval";
        case AppMode::Bench: return "bench";
    }
    return "unknown";
}

}  // namespace

App::App(AppConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.debug) {
        std::cerr << "App: mode=" << modeLabel(cfg_.mode)
                  << " camera=" << cfg_.camera_index
                  << " model=" << cfg_.model_path << "\n";
    }
}

App::~App() = default;

void App::runLive() {
    Camera camera(cfg_.camera_index);
    if (!camera.isOpen()) {
        throw std::runtime_error("App::runLive: camera failed to open");
    }

    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    while (true) {
        if (!camera.readFrame(frame)) {
            std::cerr << "App::runLive: dropped frame, retrying\n";
            continue;
        }
        cv::imshow(kWindowName, frame);
        const int key = cv::waitKey(1);
        if (key == kEscKey) {
            break;
        }
    }

    cv::destroyWindow(kWindowName);
}

void App::runEval(const std::string& /*test_dir*/,
                  const std::string& /*out_csv*/) {
    throw std::runtime_error("not implemented: App::runEval");
}

void App::runBench() {
    throw std::runtime_error("not implemented: App::runBench");
}

}  // namespace gr
