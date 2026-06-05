#pragma once

#include "gesture_types.h"

#include <deque>

namespace gd {

struct SmoothResult {
    Gesture gesture = Gesture::None;
    bool    stable  = false;  // winner holds a clear majority
};

// Majority vote over the last N gestures, to stop single frames from making
// the output jump around.
class Smoother {
public:
    explicit Smoother(std::size_t window = 8);

    // Feed one gesture per frame; get back the winner and whether it's settled.
    SmoothResult update(Gesture gesture);

    void reset();

private:
    std::size_t window_;
    std::deque<Gesture> history_;
};

}  // namespace gd
