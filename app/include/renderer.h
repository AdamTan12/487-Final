#pragma once

/// @file renderer.h
/// @brief HUD overlay (bbox, label, probability bars, FPS).

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <vector>

namespace gr {

/// Draws the live overlay on top of a webcam frame.
class Renderer {
public:
    explicit Renderer(std::vector<std::string> class_names);

    /// Draw a single frame's HUD in place.
    ///
    /// @param frame       Frame to draw on (modified in place).
    /// @param bbox        Detected hand bbox, or nullopt if no hand this frame.
    /// @param class_id    Smoothed predicted class index, or -1 if none.
    /// @param confidence  Top-1 probability for the predicted class.
    /// @param all_scores  Per-class probabilities (for the bar chart).
    /// @param fps         End-to-end frames per second.
    void draw(cv::Mat&                       frame,
              const std::optional<cv::Rect>& bbox,
              int                            class_id,
              float                          confidence,
              const std::vector<float>&      all_scores,
              double                         fps);

private:
    std::vector<std::string> class_names_;
};

}  // namespace gr
