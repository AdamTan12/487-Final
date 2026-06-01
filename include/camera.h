#pragma once

#include <opencv2/videoio.hpp>

namespace gd {

// Wraps cv::VideoCapture: picks a per-OS backend and mirrors frames so the
// preview reads like a mirror.
class Camera {
public:
    explicit Camera(int index = 0);
    ~Camera();

    Camera(const Camera&)            = delete;
    Camera& operator=(const Camera&) = delete;

    // Next frame, mirrored. False on failure (out left untouched).
    bool readFrame(cv::Mat& out);

    bool isOpen() const;

private:
    cv::VideoCapture cap_;
};

}  // namespace gd
