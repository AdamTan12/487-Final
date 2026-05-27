#include "classifier.h"

#include <stdexcept>

namespace gr {

Classifier::Classifier(const std::string& /*model_path*/) {
    // TODO: net_ = cv::dnn::readNetFromONNX(model_path);
    //       infer num_classes_ from the output layer shape.
}

std::vector<float> Classifier::classify(const cv::Mat& /*blob*/) {
    throw std::runtime_error("not implemented: Classifier::classify");
}

int Classifier::numClasses() const {
    return num_classes_;
}

}  // namespace gr
