#pragma once

#include "classical_detector.h"
#include "gesture_types.h"

#include <opencv2/core.hpp>

namespace gd {

// Debug HUD: bbox, contour, label, status panel, FPS.
class Renderer {
public:
    Renderer() = default;

    // Draws the overlay onto `frame` in place.
    void draw(cv::Mat&               frame,
              const DetectionResult& det,
              Gesture                smoothed_gesture,
              int                    finger_count,
              double                 fps,
              bool                   face_mask_on,
              bool                   shape_score_on,
              bool                   adaptive_on);

    // Frame next to the skin mask, for the `d` toggle in main.
    cv::Mat debugView(const cv::Mat&         frame,
                      const DetectionResult& det);
};

}  // namespace gd
