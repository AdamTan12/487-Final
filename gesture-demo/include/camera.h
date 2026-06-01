#pragma once

#include <opencv2/videoio.hpp>

namespace gd {

/// Thin RAII wrapper around cv::VideoCapture. Selects an OS-native backend
/// (AVFoundation on macOS, DirectShow on Windows) and mirrors frames so the
/// preview feels like a mirror.
class Camera {
public:
    explicit Camera(int index = 0);
    ~Camera();

    Camera(const Camera&)            = delete;
    Camera& operator=(const Camera&) = delete;

    /// Grab + mirror the next frame. Returns false on failure; `out` is
    /// left untouched in that case.
    bool readFrame(cv::Mat& out);

    bool isOpen() const;

private:
    cv::VideoCapture cap_;
};

}  // namespace gd
