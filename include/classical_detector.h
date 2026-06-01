#pragma once

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <vector>

namespace gd {

struct DetectionResult {
    bool                   found = false;
    std::vector<cv::Point> contour;
    cv::Rect               bbox;
    cv::Mat                mask;            ///< Final binary mask after all processing.
    std::vector<cv::Rect>  faces;           ///< Cascade hits this frame (debug overlay).
    bool                   adaptive_active = false;  ///< True if back-projection was used.
    int                    trained_frames  = 0;      ///< Number of face samples accumulated.
};

/// HSV skin detector with three runtime-toggleable add-ons:
///   * Face cascade (`f`) — zero detected faces in the mask before contours.
///   * Shape scoring (`s`) — pick by `area * (1 - solidity)^k`, not pure area.
///   * Adaptive skin model (`a`) — learn the user's H-S distribution from
///     the face patch and use `calcBackProject` for a calibrated mask.
///
/// The face cascade runs every Nth frame and the result is cached, which
/// keeps FPS up.
class ClassicalDetector {
public:
    ClassicalDetector();

    DetectionResult detect(const cv::Mat& frame);

    // --- runtime knobs ----------------------------------------------------
    int  v_min             = 60;
    bool use_face_mask     = true;
    bool use_shape_score   = true;
    bool use_adaptive_skin = true;

private:
    cv::CascadeClassifier face_cascade_;

    // Cached cascade state so detection doesn't have to run every frame.
    std::vector<cv::Rect> cached_faces_;
    int                   detect_calls_ = 0;

    // Adaptive skin model.
    cv::Mat skin_hist_;       ///< 2D H-S histogram, EMA-updated from face patches.
    int     trained_frames_ = 0;
};

}  // namespace gd
