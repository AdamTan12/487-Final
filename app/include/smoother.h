#pragma once

/// @file smoother.h
/// @brief Temporal smoothing of per-frame class scores.

#include <deque>
#include <vector>

namespace gr {

/// Averages the last N softmax vectors and argmaxes the result.
///
/// Eliminates single-frame flicker and one-frame misclassifications
/// without adding noticeable latency at typical webcam framerates.
/// Window size of 5 (~150 ms at 30 fps) is a reasonable default.
class Smoother {
public:
    explicit Smoother(int window_size = 5);

    /// Push a fresh score vector, return the smoothed argmax class id.
    int update(const std::vector<float>& scores);

    /// Drop history (e.g., after a detection gap).
    void reset();

private:
    int                              window_size_;
    std::deque<std::vector<float>>   history_;
};

}  // namespace gr
