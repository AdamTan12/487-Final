#pragma once

#include "classical_detector.h"

#include <opencv2/core.hpp>

#include <string>

namespace gd {

/// HUD overlay: bbox + contour + label tag + status panel + FPS.
class Renderer {
public:
    Renderer() = default;

    /// Draw the per-frame overlay in place on `frame`.
    void draw(cv::Mat&               frame,
              const DetectionResult& det,
              const std::string&     smoothed_label,
              int                    finger_count,
              double                 fps,
              bool                   face_mask_on,
              bool                   shape_score_on,
              bool                   adaptive_on);

    /// Returns `frame` side-by-side with the (BGR-promoted) skin mask.
    /// Useful for the `d` (debug) toggle in main.
    cv::Mat debugView(const cv::Mat&         frame,
                      const DetectionResult& det);
};

}  // namespace gd
