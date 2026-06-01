#include "classical_classifier.h"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <limits>

namespace gd {

namespace {

// --- tunable thresholds -----------------------------------------------------
// kDefectDepthMin moved to ClassicalClassifier::defect_depth_min so it can be
// adjusted live from the demo with [ and ].
constexpr double kDefectAngleMax = 3.14159265358979 / 2.0;  ///< 90° in radians

/// Reject defects whose start/end ("fingertips") aren't measurably farther
/// from the contour centroid than the far point. This kills wrist/forearm
/// defects (which have the opposite geometry — far is the most distal
/// point, not start/end). Set > 1.0 for a stricter margin.
constexpr double kFingertipMarginFactor = 1.0;

constexpr std::size_t kMinHullPoints = 3;
// ---------------------------------------------------------------------------

constexpr std::array<const char*, 6> kLabelByFingerCount = {
    "fist",   // 0
    "point",  // 1
    "peace",  // 2
    "three",  // 3
    "four",   // 4
    "palm",   // 5
};

/// Angle at point `far` formed by the triangle (`start`, `end`, `far`),
/// using the law of cosines on the side lengths.
double angleAtFar(const cv::Point& start, const cv::Point& end, const cv::Point& far) {
    const double a = cv::norm(start - end);
    const double b = cv::norm(far   - start);
    const double c = cv::norm(far   - end);
    const double denom = 2.0 * b * c;
    if (denom <= 0.0) {
        return std::numeric_limits<double>::infinity();  // skip degenerate
    }
    double cos_angle = (b * b + c * c - a * a) / denom;
    if (cos_angle >  1.0) cos_angle =  1.0;
    if (cos_angle < -1.0) cos_angle = -1.0;
    return std::acos(cos_angle);
}

const char* labelFor(int finger_count) {
    if (finger_count < 0) return "none";
    if (finger_count >= static_cast<int>(kLabelByFingerCount.size())) {
        return "palm";   // anything beyond our table reads as an open hand
    }
    return kLabelByFingerCount[static_cast<std::size_t>(finger_count)];
}

}  // namespace

ClassificationResult ClassicalClassifier::classify(
        const std::vector<cv::Point>& contour) {
    ClassificationResult result;
    if (contour.size() < kMinHullPoints) {
        return result;
    }

    std::vector<int> hull_indices;
    cv::convexHull(contour, hull_indices, /*clockwise=*/false, /*returnPoints=*/false);
    if (hull_indices.size() < kMinHullPoints) {
        return result;
    }

    std::vector<cv::Vec4i> defects;
    try {
        cv::convexityDefects(contour, hull_indices, defects);
    } catch (const cv::Exception&) {
        // convexityDefects throws if the contour isn't cleanly
        // convex-decomposable. Treat that as "no usable hand this frame".
        return result;
    }
    if (defects.empty()) {
        return result;
    }

    // Contour centroid via image moments — used to reject wrist-style defects.
    const cv::Moments m = cv::moments(contour);
    if (m.m00 <= 0.0) {
        return result;
    }
    const cv::Point2f centroid(static_cast<float>(m.m10 / m.m00),
                               static_cast<float>(m.m01 / m.m00));

    int gaps = 0;
    for (const auto& d : defects) {
        const cv::Point start = contour[static_cast<std::size_t>(d[0])];
        const cv::Point end   = contour[static_cast<std::size_t>(d[1])];
        const cv::Point far   = contour[static_cast<std::size_t>(d[2])];
        const double    depth = d[3] / 256.0;
        if (depth <= defect_depth_min) {
            continue;
        }
        const double angle = angleAtFar(start, end, far);
        if (angle > kDefectAngleMax) {
            continue;
        }

        // Wrist-defect rejection: both "fingertips" must be farther from
        // the centroid than the valley's far point.
        const double d_start_c = cv::norm(cv::Point2f(start) - centroid);
        const double d_end_c   = cv::norm(cv::Point2f(end)   - centroid);
        const double d_far_c   = cv::norm(cv::Point2f(far)   - centroid);
        if (d_start_c <= d_far_c * kFingertipMarginFactor) continue;
        if (d_end_c   <= d_far_c * kFingertipMarginFactor) continue;

        ++gaps;
    }

    result.finger_count = (gaps > 0) ? gaps + 1 : 0;
    result.label        = labelFor(result.finger_count);
    return result;
}

}  // namespace gd
