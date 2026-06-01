#include "smoother.h"

#include <unordered_map>

namespace gd {

namespace {
constexpr std::size_t kDefaultWindow = 8;
}  // namespace

Smoother::Smoother(std::size_t window)
    : window_(window == 0 ? kDefaultWindow : window) {}

std::string Smoother::update(const std::string& label) {
    history_.push_back(label);
    if (history_.size() > window_) {
        history_.pop_front();
    }

    std::unordered_map<std::string, int> counts;
    counts.reserve(history_.size());
    for (const auto& l : history_) {
        ++counts[l];
    }

    const std::string* best = &history_.back();
    int                best_count = 0;
    for (const auto& [name, count] : counts) {
        if (count > best_count) {
            best_count = count;
            best       = &name;
        }
    }
    return *best;
}

void Smoother::reset() {
    history_.clear();
}

}  // namespace gd
