#pragma once

#include <opencv2/core.hpp>

// Types shared between the CV side and the display side. This is the only
// header both sides include to talk to each other, so keep it small.

namespace gd {

// What we recognize. To add one: a value here + an emoji on the display side.
enum class Gesture {
    None,
    Fist,     // 0 fingers
    Peace,    // 2 fingers
    Three,    // 3 fingers
    Four,     // 4 fingers
    Palm,     // 5 fingers
};

// One frame's result, handed across to draw_gesture().
struct GestureEvent {
    Gesture  gesture      = Gesture::None;
    bool     stable       = false;          // survived smoothing
    cv::Rect bbox;                          // where to draw the emoji
    int      finger_count = 0;              // debug only
};

// Name for logs / the debug HUD. Not the emoji.
inline const char* gesture_name(Gesture g) {
    switch (g) {
        case Gesture::None:    return "none";
        case Gesture::Fist:    return "fist";
        case Gesture::Peace:   return "peace";
        case Gesture::Three:   return "three";
        case Gesture::Four:    return "four";
        case Gesture::Palm:    return "palm";
    }
    return "none";
}

}  // namespace gd
