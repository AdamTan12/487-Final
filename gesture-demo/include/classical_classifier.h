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
};

}  // namespace gd
