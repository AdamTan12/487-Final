#pragma once

/// @file classical_detector.h
/// @brief HSV + contours hand detector. Primary detector strategy.

#include "i_hand_detector.h"

#include <opencv2/core.hpp>

namespace gr {

/// Thresholds and shape parameters for the classical pipeline.
struct ClassicalDetectorConfig {
    cv::Scalar hsv_low         = cv::Scalar(0, 30, 60);    ///< Lower skin-tone HSV bound
    cv::Scalar hsv_high        = cv::Scalar(20, 150, 255); ///< Upper skin-tone HSV bound
    int        morph_kernel    = 5;                        ///< Square kernel for open/close (px)
    double     min_contour_area = 5000.0;                  ///< Reject blobs smaller than this (px^2)
    float      padding_ratio    = 0.15f;                   ///< Expand the bbox by this fraction
};

/// HSV-threshold-based hand detector.
///
/// Pipeline:
///   1. Convert BGR -> HSV.
///   2. Threshold against skin-tone HSV bounds.
///   3. Morphological open + close to clean the mask.
///   4. Find external contours, pick the largest by area.
///   5. Compute its bounding box, pad by `padding_ratio`, clamp to frame.
///
/// This is fast and dependency-free but sensitive to lighting and
/// background skin tones. Falls back to OnnxDetector when accuracy
/// matters more than latency.
class ClassicalDetector : public IHandDetector {
public:
    explicit ClassicalDetector(const ClassicalDetectorConfig& cfg = {});

    std::optional<cv::Rect> detect(const cv::Mat& frame) override;

private:
    ClassicalDetectorConfig cfg_;
};

}  // namespace gr
