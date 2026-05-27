#include "smoother.h"

#include <stdexcept>

namespace gr {

Smoother::Smoother(int window_size) : window_size_(window_size) {}

int Smoother::update(const std::vector<float>& /*scores*/) {
    throw std::runtime_error("not implemented: Smoother::update");
}

void Smoother::reset() {
    history_.clear();
}

}  // namespace gr
