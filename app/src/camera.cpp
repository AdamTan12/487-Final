#include "camera.h"

#include <opencv2/videoio.hpp>

#include <stdexcept>
#include <string>

namespace gr {

namespace {

int pickDefaultBackend() {
#if defined(__APPLE__)
    return cv::CAP_AVFOUNDATION;
#elif defined(_WIN32)
    // MSMF is the modern Windows backend; DSHOW is a workable fallback if MSMF
    // misbehaves (some USB cams expose timing quirks under MSMF).
    return cv::CAP_MSMF;
#else
    return cv::CAP_V4L2;
#endif
}

}  // namespace

Camera::Camera(int index, int backend_hint) {
    const int backend = (backend_hint == cv::CAP_ANY) ? pickDefaultBackend()
                                                      : backend_hint;
    if (!cap_.open(index, backend)) {
        throw std::runtime_error("Camera: failed to open device index "
                                 + std::to_string(index));
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
    out = frame;
    return true;
}

bool Camera::isOpen() const {
    return cap_.isOpened();
}

}  // namespace gr
