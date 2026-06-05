#pragma once

#include <opencv2/videoio.hpp>

namespace gd {

/// Thin RAII wrapper around cv::VideoCapture.
/// Selects the best OS-native backend (AVFoundation on macOS, DirectShow on
/// Windows, auto elsewhere) and horizontally flips every frame so the live
/// preview reads like a mirror.
class Camera {
public:
    /// Opens the camera at `index`. Prints a warning to stderr on failure;
    /// check isOpen() before calling readFrame().
    explicit Camera(int index = 0);

    /// Releases the underlying capture device.
    ~Camera();

    Camera(const Camera&)            = delete;
    Camera& operator=(const Camera&) = delete;

    /// Captures the next frame into `out`, flipped horizontally.
    /// Returns false if the device is closed or the read fails; `out` is
    /// left unmodified on failure.
    bool readFrame(cv::Mat& out);

    /// Returns true if the underlying capture device opened successfully.
    bool isOpen() const;

private:
    cv::VideoCapture cap_;
};

}  // namespace gd
