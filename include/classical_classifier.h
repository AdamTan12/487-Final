#pragma once

#include "gesture_types.h"

#include <opencv2/core.hpp>

#include <vector>

namespace gd {

struct ClassificationResult {
    Gesture gesture      = Gesture::None;
    int     finger_count = 0;
};

// Counts extended fingers from the hand contour via convexity defects, then
// maps the count to a Gesture. No model, just geometry.
class ClassicalClassifier {
public:
    ClassicalClassifier() = default;

    // Safe to call on tiny or degenerate contours. `has_hole` (an enclosed
    // gap, e.g. the OK loop) overrides the finger count when set.
    ClassificationResult classify(const std::vector<cv::Point>& contour,
                                  bool has_hole = false);

    // Min defect depth (px) for a gap to count as one between fingers.
    // Higher rejects knuckle/thumb wrinkles. Tune live with [ and ].
    double defect_depth_min = 20.0;

    // When the finger count is ambiguous (0-1), split fist from open palm by
    // solidity: below this is an open hand, above it is a fist. Tune with o/p.
    double open_solidity_max = 0.85;
};

}  // namespace gd
