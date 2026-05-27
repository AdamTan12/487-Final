#pragma once

/// @file i_hand_detector.h
/// @brief Common interface for any hand-localization strategy.

#include <opencv2/core.hpp>

#include <optional>

namespace gr {

/// Abstract base for hand detectors. Implementations are interchangeable
/// at runtime via the `--detector` CLI flag.
///
/// @note The returned rectangle is expressed in full-frame pixel
///       coordinates (not normalized, not relative to a crop). Callers
///       can pass it directly to cv::Mat::operator() / cv::Rect crops.
class IHandDetector {
public:
    virtual ~IHandDetector() = default;

    /// Locate a single hand in `frame`. Returns std::nullopt when no
    /// hand is found with sufficient confidence.
    virtual std::optional<cv::Rect> detect(const cv::Mat& frame) = 0;
};

}  // namespace gr
