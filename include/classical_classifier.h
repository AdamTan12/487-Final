#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace gd {

struct ClassificationResult {
    std::string label        = "none";
    int         finger_count = 0;
};

/// Classical "classifier": counts extended fingers using the convexity-defect
/// trick (Bradski / OpenCV tutorial), then maps the count to a label name.
/// No training data, no model file — pure geometry.
class ClassicalClassifier {
public:
    ClassicalClassifier() = default;

    /// Classify a hand contour. Safe on small / degenerate contours.
    ClassificationResult classify(const std::vector<cv::Point>& contour);

    /// Minimum convexity-defect depth (px) to count as a finger gap.
    /// Raise to reject thumb/knuckle wrinkles; lower to be more sensitive.
    /// Live-tunable via [ and ] in the demo.
    double defect_depth_min = 20.0;
};

}  // namespace gd
