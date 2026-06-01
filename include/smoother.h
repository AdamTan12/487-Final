#pragma once

#include <deque>
#include <string>

namespace gd {

/// Sliding-window majority-vote smoother over discrete labels. Eliminates
/// single-frame flicker without adding noticeable latency.
class Smoother {
public:
    explicit Smoother(std::size_t window = 8);

    /// Push a new label, return the most frequent label in the window.
    std::string update(const std::string& label);

    void reset();

private:
    std::size_t            window_;
    std::deque<std::string> history_;
};

}  // namespace gd
