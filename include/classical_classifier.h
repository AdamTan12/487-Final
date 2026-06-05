#pragma once

#include "gesture_types.h"

#include <opencv2/core.hpp>

#include <vector>

namespace gd {

/// Output of ClassicalClassifier::classify() for a single hand contour.
struct ClassificationResult {
    Gesture gesture      = Gesture::None; ///< Recognised gesture (or None if classification failed).
    int     finger_count = 0;             ///< Number of extended fingers inferred from convexity defects.
};

/// Geometry-only hand classifier — no trained model.
///
/// Counts inter-finger gaps (convexity defects) on the hand contour and maps
/// the count to a Gesture enum value. The OK sign is a special case: it is
/// detected as a topological hole (thumb+index loop) rather than by finger
/// count, so `has_hole` must be passed from the detector.
class ClassicalClassifier {
public:
    ClassicalClassifier() = default;

    /// Classifies a hand contour into a Gesture.
    /// Safe to call on tiny or degenerate contours; returns Gesture::None in
    /// those cases. When `has_hole` is true and solidity indicates an open
    /// hand, the result is Gesture::Ok regardless of finger count.
    ClassificationResult classify(const std::vector<cv::Point>& contour,
                                  bool has_hole = false);

    /// Minimum convexity-defect depth (pixels) for a gap to count as a space
    /// between two fingers. Increase to reject knuckle/thumb-base wrinkles.
    /// Tunable at runtime with the [ and ] keys.
    double defect_depth_min = 20.0;

    /// Solidity threshold used to disambiguate a fist from an open palm when
    /// the finger count is 0 or 1. Contours with solidity below this value are
    /// classified as Palm; at or above it as Fist.
    /// Tunable at runtime with the o and p keys.
    double open_solidity_max = 0.85;
};

}  // namespace gd
