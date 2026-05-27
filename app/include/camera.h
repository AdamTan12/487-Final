#pragma once

/// @file camera.h
/// @brief Thin RAII wrapper over cv::VideoCapture with platform-aware backend.

#include <opencv2/videoio.hpp>

namespace gr {

/// Opens a camera and pulls frames. The backend defaults to the OS-native
/// one (AVFoundation on macOS, DSHOW/MSMF on Windows, V4L2 on Linux) but
/// callers can override via `backend_hint`.
///
/// The capture is released by the destructor.
class Camera {
public:
    explicit Camera(int index = 0, int backend_hint = cv::CAP_ANY);
    ~Camera();

    Camera(const Camera&)            = delete;
    Camera& operator=(const Camera&) = delete;

    /// Grab the next frame. Returns false if the camera has dropped or
    /// the device is unreachable. `out` is left untouched on failure.
    bool readFrame(cv::Mat& out);

    /// True if the underlying capture is open and ready.
    bool isOpen() const;

private:
    cv::VideoCapture cap_;
};

}  // namespace gr
