#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace gd {

/// Output of a single detection pass.
struct DetectionResult {
    bool                   found = false;
    std::vector<cv::Point> contour;
    cv::Rect               bbox;
    cv::Mat                mask;   ///< Binary skin mask, kept for debug overlay.
};

/// HSV skin-segmentation hand detector.
///
/// Pipeline: BGR → HSV → inRange → morphological open + close → optional
/// Gaussian blur → external contours → largest above min-area. The HSV
/// V-min (`v_min`) is intentionally public so the demo can tweak it live
/// with +/- keys.
class ClassicalDetector {
public:
    ClassicalDetector() = default;

    DetectionResult detect(const cv::Mat& frame);

    /// Lower V (value) bound on the HSV skin mask. Higher → reject darker
    /// shadows; lower → admit more skin under dim light.
    int v_min = 60;
};

}  // namespace gd
