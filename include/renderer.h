#pragma once

#include "classical_detector.h"
#include "gesture_types.h"

#include <opencv2/core.hpp>

namespace gd {

/// Draws the debug HUD directly onto a live camera frame.
/// Renders the hand bounding box, contour outline, gesture label, a
/// semi-transparent status panel (gesture/fingers/FPS/toggle states),
/// and optionally a side-by-side debug view of the skin mask.
class Renderer {
public:
    Renderer() = default;

    /// Composites the full debug overlay onto `frame` in place.
    /// `det` provides the contour and bbox; the remaining arguments supply
    /// the values shown in the status panel.
    void draw(cv::Mat&               frame,
              const DetectionResult& det,
              Gesture                smoothed_gesture,
              int                    finger_count,
              double                 fps,
              bool                   face_mask_on,
              bool                   shape_score_on,
              bool                   adaptive_on);

    /// Returns a new image that places the current frame (with red boxes
    /// highlighting detected faces) side-by-side with the binary skin mask.
    /// Used when the `d` key debug view is active.
    cv::Mat debugView(const cv::Mat&         frame,
                      const DetectionResult& det);
};

}  // namespace gd
