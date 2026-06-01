#pragma once

#include "gesture_types.h"

#include <deque>

namespace gd {

/// Result of one smoothing step.
struct SmoothResult {
    Gesture gesture = Gesture::None;
    bool    stable  = false;  ///< true once the winner holds a strong majority
};

/// Sliding-window majority-vote smoother over discrete gestures. Eliminates
/// single-frame flicker without adding noticeable latency, and reports whether
/// the current winner is stable enough to display.
class Smoother {
public:
    explicit Smoother(std::size_t window = 8);

    /// Push a new gesture, return the smoothed winner + stability flag.
    SmoothResult update(Gesture gesture);

    void reset();

private:
    std::size_t          window_;
    std::deque<Gesture>  history_;
};

}  // namespace gd
