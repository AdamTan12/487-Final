#include "smoother.h"

#include <array>

namespace gd {

namespace {
constexpr std::size_t kDefaultWindow = 8;

// How much of the window the winner needs before we call it stable.
constexpr double kStableFraction = 0.6;

// Size of the Gesture enum. Keep in sync if you add gestures.
constexpr std::size_t kGestureCount = 6;
}  // namespace

Smoother::Smoother(std::size_t window)
    : window_(window == 0 ? kDefaultWindow : window) {}

SmoothResult Smoother::update(Gesture gesture) {
    history_.push_back(gesture);
    if (history_.size() > window_) {
        history_.pop_front();
    }

    std::array<int, kGestureCount> counts{};
    for (const Gesture g : history_) {
        ++counts[static_cast<std::size_t>(g)];
    }

    std::size_t best_idx   = static_cast<std::size_t>(history_.back());
    int         best_count = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] > best_count) {
            best_count = counts[i];
            best_idx   = i;
        }
    }

    SmoothResult result;
    result.gesture = static_cast<Gesture>(best_idx);
    result.stable  = result.gesture != Gesture::None &&
                     best_count >= static_cast<int>(window_ * kStableFraction);
    return result;
}

void Smoother::reset() {
    history_.clear();
}

}  // namespace gd
