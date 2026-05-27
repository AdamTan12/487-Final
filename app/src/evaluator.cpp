#include "evaluator.h"

#include "app.h"

#include <stdexcept>

namespace gr {

Evaluator::Evaluator(App& app) : app_(app) {}

void Evaluator::evaluate(const std::string& /*test_dir*/,
                         const std::string& /*output_csv*/) {
    throw std::runtime_error("not implemented: Evaluator::evaluate");
}

}  // namespace gr
