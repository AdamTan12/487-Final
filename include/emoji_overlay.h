#pragma once

#include "gesture_types.h"

#include <opencv2/core.hpp>

namespace gd {

// =============================================================================
//  DISPLAY LAYER
//
//  Single entry point for the emoji overlay. The CV pipeline produces a
//  GestureEvent each frame and passes it here; this function handles
//  everything on the display side: asset loading, sizing, placement,
//  alpha-blending, and the fade-in animation.
//
//  Emoji PNGs (72×72 BGRA Twemoji) live in assets/emoji/ relative to the
//  working directory. They are loaded lazily on first use and cached.
//  A missing PNG prints one warning and silently skips the overlay for that
//  gesture; the function never throws.
// =============================================================================

/// Draws the emoji matching `ev.gesture` onto `frame` in place.
/// The emoji is sized proportionally to `ev.bbox`, centred above the box, and
/// eased in over several frames once `ev.stable` becomes true.
/// No-ops when `ev.gesture` is None, `ev.stable` is false, or the frame is
/// not a BGR 8-bit image.
void draw_gesture(cv::Mat& frame, const GestureEvent& ev);

}  // namespace gd
