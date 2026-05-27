#include "classical_detector.h"

#include <stdexcept>

namespace gr {

ClassicalDetector::ClassicalDetector(const ClassicalDetectorConfig& cfg)
    : cfg_(cfg) {}

std::optional<cv::Rect> ClassicalDetector::detect(const cv::Mat& /*frame*/) {
    throw std::runtime_error("not implemented: ClassicalDetector::detect");
}

}  // namespace gr
