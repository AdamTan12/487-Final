#include "classical_classifier.h"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <limits>

namespace gd {

namespace {

// (defect depth lives on the class so it can be tuned live; see the header.)
constexpr double kDefectAngleMax = 3.14159265358979 / 2.0;  // 90 degrees

// A real finger gap has its fingertips farther from the hand centre than the
// valley between them. Wrist/forearm defects are the other way round, so this
// throws them out. Bump above 1.0 to be stricter.
constexpr double kFingertipMarginFactor = 1.0;

constexpr std::size_t kMinHullPoints = 3;

// finger count -> gesture. Defects count the gaps *between* fingers, so a
// single finger never registers (min non-zero count is 2). Count 1 can't
// happen and just maps to None.
constexpr std::array<Gesture, 6> kGestureByFingerCount = {
    Gesture::Fist,   // 0
    Gesture::None,   // 1  (can't happen)
    Gesture::Peace,  // 2
    Gesture::Three,  // 3
    Gesture::Four,   // 4
    Gesture::Palm,   // 5
};

/// Computes the angle at vertex `far` in the triangle formed by (start, end, far)
/// using the law of cosines. Returns infinity for degenerate triangles so the
/// caller can skip them without special-casing NaN.
double angleAtFar(const cv::Point& start, const cv::Point& end, const cv::Point& far) {
    const double a = cv::norm(start - end);
    const double b = cv::norm(far   - start);
    const double c = cv::norm(far   - end);
    const double denom = 2.0 * b * c;
    if (denom <= 0.0) {
        return std::numeric_limits<double>::infinity();  // degenerate, skip it
    }
    double cos_angle = (b * b + c * c - a * a) / denom;
    if (cos_angle >  1.0) cos_angle =  1.0;
    if (cos_angle < -1.0) cos_angle = -1.0;
    return std::acos(cos_angle);
}

/// Maps a finger count to the corresponding Gesture value.
/// Counts beyond 5 are clamped to Palm; negative counts return None.
Gesture gestureFor(int finger_count) {
    if (finger_count < 0) return Gesture::None;
    if (finger_count >= static_cast<int>(kGestureByFingerCount.size())) {
        return Gesture::Palm;   // more than 5 -> treat as open hand
    }
    return kGestureByFingerCount[static_cast<std::size_t>(finger_count)];
}

}  // namespace

ClassificationResult ClassicalClassifier::classify(
        const std::vector<cv::Point>& contour, bool has_hole) {
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
        // convexityDefects throws on contours it can't decompose; bail.
        return result;
    }
    if (defects.empty()) {
        return result;
    }

    // Centroid (image moments), needed for the wrist-defect check below.
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

        // Both fingertips have to sit farther out than the valley, or it's
        // a wrist gap, not a finger gap.
        const double d_start_c = cv::norm(cv::Point2f(start) - centroid);
        const double d_end_c   = cv::norm(cv::Point2f(end)   - centroid);
        const double d_far_c   = cv::norm(cv::Point2f(far)   - centroid);
        if (d_start_c <= d_far_c * kFingertipMarginFactor) continue;
        if (d_end_c   <= d_far_c * kFingertipMarginFactor) continue;

        ++gaps;
    }

    const int fingers = (gaps > 0) ? gaps + 1 : 0;
    result.finger_count = fingers;

    // Solidity = how much of its convex hull the contour fills. A fist is
    // compact (high); an open hand leaves gaps (low). Used both to gate the OK
    // sign (a fist's shadow can look like a hole) and to split fist from palm.
    std::vector<cv::Point> hull_pts;
    cv::convexHull(contour, hull_pts);
    const double hull_area = cv::contourArea(hull_pts);
    const double area      = cv::contourArea(contour);
    const double solidity  = (hull_area > 0.0) ? area / hull_area : 1.0;
    const bool   open_hand = solidity < open_solidity_max;

    if (has_hole && open_hand) {
        result.gesture = Gesture::Ok;     // a loop, and the hand isn't a fist
    } else if (fingers >= 2) {
        result.gesture = gestureFor(fingers);
    } else {
        result.gesture = open_hand ? Gesture::Palm : Gesture::Fist;
    }
    return result;
}

}  // namespace gd
