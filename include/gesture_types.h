#pragma once

#include <opencv2/core.hpp>

// Shared contract between the CV pipeline and the display layer.
// This is the only header both sides include, so keep it minimal.

namespace gd {

/// All gesture classes the pipeline can recognise.
/// To add a new gesture: append a value here and provide a matching emoji asset.
enum class Gesture {
    None,
    Fist,     // 0 fingers
    Peace,    // 2 fingers
    Three,    // 3 fingers
    Four,     // 4 fingers
    Palm,     // 5 fingers
    Ok,       // thumb+index loop (detected as a hole, not by finger count)
};

/// Per-frame result passed from the CV pipeline to draw_gesture().
struct GestureEvent {
    Gesture  gesture      = Gesture::None; ///< Smoothed gesture for this frame.
    bool     stable       = false;         ///< True once the gesture holds a clear majority in the smoother window.
    cv::Rect bbox;                         ///< Bounding box of the detected hand, used to position the emoji.
    int      finger_count = 0;             ///< Raw finger count from the classifier (debug display only).
};

/// Returns a human-readable name for `g`, used in the debug HUD and logs.
/// Returns "none" for any unrecognised value.
inline const char* gesture_name(Gesture g) {
    switch (g) {
        case Gesture::None:    return "none";
        case Gesture::Fist:    return "fist";
        case Gesture::Peace:   return "peace";
        case Gesture::Three:   return "three";
        case Gesture::Four:    return "four";
        case Gesture::Palm:    return "palm";
        case Gesture::Ok:      return "ok";
    }
    return "none";
}

}  // namespace gd
