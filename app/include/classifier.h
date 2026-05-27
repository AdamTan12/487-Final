#pragma once

/// @file classifier.h
/// @brief Gesture classifier (MobileNetV3-small fine-tune, ONNX).

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

namespace gr {

/// Wraps the gesture-classification ONNX model.
///
/// The input blob format is fixed by the training pipeline; see
/// `preprocess.h` and `models/preprocessing.txt`. Output is per-class
/// scores (softmax already applied in-graph, or applied here if the
/// graph emits logits — implementation will normalize either way).
class Classifier {
public:
    explicit Classifier(const std::string& model_path);

    /// Run a forward pass on a pre-built blob.
    /// @param blob 1x3xHxW float blob, normalized per `preprocessing.txt`.
    /// @return Vector of per-class probabilities, length == numClasses().
    std::vector<float> classify(const cv::Mat& blob);

    int numClasses() const;

private:
    cv::dnn::Net net_;
    int          num_classes_ = 0;
};

}  // namespace gr
