#pragma once

#include <opencv2/core.hpp>

// =============================================================================
//  SHARED CONTRACT between the CV pipeline and the display/emoji layer.
//
//  This header is the agreed boundary. Both sides include ONLY this file to
//  talk to each other. Change it only by mutual agreement — adding a gesture
//  means adding one enum value (CV side detects it) plus one emoji asset
//  (display side renders it); no function signatures change.
// =============================================================================

namespace gd {

/// The full vocabulary of gestures the demo knows about. The CV pipeline
/// produces these; the display layer maps each to an emoji. A value may exist
/// here before the detector can reliably produce it — that's expected.
enum class Gesture {
    None,     // no hand / not classifiable
    Fist,     // ✊  0 fingers
    Like,     // 👍  1 finger, pointing up   (thumb up)
    Dislike,  // 👎  1 finger, pointing down (thumb down)
    Peace,    // ✌️  2 fingers
    Three,    // 🤟  3 fingers
    Four,     // 🤚  4 fingers
    Palm,     // 🖐️  5 fingers
};

/// Everything the display layer needs for one frame. Produced by the CV
/// pipeline, consumed by draw_gesture(). The display side never sees a contour
/// or a mask; the CV side never sees an emoji.
struct GestureEvent {
    Gesture  gesture      = Gesture::None;  ///< what to show
    bool     stable       = false;          ///< held long enough to trust/display
    cv::Rect bbox;                          ///< hand location, for emoji placement
    int      finger_count = 0;              ///< raw count, for debug overlay only
};

/// Human-readable name for logging / debug HUD. This is NOT the emoji — the
/// emoji mapping lives entirely on the display side.
inline const char* gesture_name(Gesture g) {
    switch (g) {
        case Gesture::None:    return "none";
        case Gesture::Fist:    return "fist";
        case Gesture::Like:    return "like";
        case Gesture::Dislike: return "dislike";
        case Gesture::Peace:   return "peace";
        case Gesture::Three:   return "three";
        case Gesture::Four:    return "four";
        case Gesture::Palm:    return "palm";
    }
    return "none";
}

}  // namespace gd
