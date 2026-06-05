#include "camera.h"

#include <opencv2/imgproc.hpp>

#include <iostream>

namespace gd {

namespace {

/// Returns the best cv::VideoCapture backend for the current platform.
/// Prefer native APIs over the generic fallback to avoid capture delays and
/// permission issues (e.g. AVFoundation is required for macOS camera access).
int pickBackend() {
#if defined(__APPLE__)
    return cv::CAP_AVFOUNDATION;
#elif defined(_WIN32)
    return cv::CAP_DSHOW;
#else
    return cv::CAP_ANY;
#endif
}

}  // namespace

Camera::Camera(int index) {
    if (!cap_.open(index, pickBackend())) {
        std::cerr << "Camera: failed to open device index " << index
                  << " (check OS camera permission for this terminal app)\n";
    }
}

Camera::~Camera() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool Camera::readFrame(cv::Mat& out) {
    if (!cap_.isOpened()) {
        return false;
    }
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
        return false;
    }
    cv::flip(frame, out, /*flipCode=*/1);  // horizontal mirror
    return true;
}

bool Camera::isOpen() const {
    return cap_.isOpened();
}

}  // namespace gd
