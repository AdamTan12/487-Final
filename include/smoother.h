#pragma once

#include "gesture_types.h"

#include <deque>

namespace gd {

/// Output of Smoother::update() for a single frame.
struct SmoothResult {
    Gesture gesture = Gesture::None; ///< Gesture with the highest vote count in the current window.
    bool    stable  = false;         ///< True when the winner holds at least 60 % of the window.
};

/// Temporal majority-vote filter over a sliding window of raw gesture frames.
/// Prevents single-frame misclassifications from flickering through to the UI.
class Smoother {
public:
    /// Constructs a smoother with a history window of `window` frames.
    /// Passing 0 uses the default window size of 8.
    explicit Smoother(std::size_t window = 8);

    /// Appends `gesture` to the history, drops the oldest entry if the window
    /// is full, and returns the current winner with its stability flag.
    SmoothResult update(Gesture gesture);

    /// Clears the history, resetting all state as if freshly constructed.
    void reset();

private:
    std::size_t window_;
    std::deque<Gesture> history_;
};

}  // namespace gd
