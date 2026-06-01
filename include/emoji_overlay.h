#pragma once

#include "gesture_types.h"

#include <opencv2/core.hpp>

namespace gd {

// =============================================================================
//  DISPLAY LAYER  (teammate-owned)
//
//  This is the entire surface the display side exposes to the rest of the app.
//  The CV pipeline hands you a GestureEvent each frame; you draw the matching
//  emoji onto `frame` in place. You own everything behind this function:
//  loading emoji assets, the Gesture->asset table, placement, alpha-blending,
//  fade-in on `stable`, etc. You never need to touch the CV code.
//
//  To develop in parallel, drive this with a mock producer (see the demo
//  harness note in emoji_overlay.cpp) — you do not need the real detector to
//  build and test the overlay.
// =============================================================================

/// Draw the emoji for `ev` onto `frame`, in place. A no-op if there is nothing
/// worth showing (gesture None, or not yet stable — your call).
void draw_gesture(cv::Mat& frame, const GestureEvent& ev);

}  // namespace gd
