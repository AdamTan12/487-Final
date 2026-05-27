#include "renderer.h"

#include <stdexcept>
#include <utility>

namespace gr {

Renderer::Renderer(std::vector<std::string> class_names)
    : class_names_(std::move(class_names)) {}

void Renderer::draw(cv::Mat&                       /*frame*/,
                    const std::optional<cv::Rect>& /*bbox*/,
                    int                            /*class_id*/,
                    float                          /*confidence*/,
                    const std::vector<float>&      /*all_scores*/,
                    double                         /*fps*/) {
    throw std::runtime_error("not implemented: Renderer::draw");
}

}  // namespace gr
