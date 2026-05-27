#include "onnx_detector.h"

#include <stdexcept>

namespace gr {

OnnxDetector::OnnxDetector(const std::string& /*model_path*/,
                           const cv::Size&    input_size,
                           float              conf_threshold,
                           float              nms_threshold)
    : input_size_(input_size),
      conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold) {
    // TODO: net_ = cv::dnn::readNetFromONNX(model_path);
}

std::optional<cv::Rect> OnnxDetector::detect(const cv::Mat& /*frame*/) {
    throw std::runtime_error("not implemented: OnnxDetector::detect");
}

}  // namespace gr
