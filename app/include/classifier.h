#pragma once

/// @file classifier.h
/// @brief Gesture classifier (HaGRID pretrained ResNet18, ONNX).

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace gr {

/// Wraps the gesture-classification ONNX model.
///
/// Inference runs through Microsoft's onnxruntime (not cv::dnn) so the
/// numerical behavior matches the Python smoke test exactly. The input
/// blob format is fixed by the training pipeline; see `preprocess.h`
/// and `models/preprocessing.txt`. Output is per-class probabilities
/// (softmax applied here -- the ONNX graph emits raw logits).
class Classifier {
public:
    explicit Classifier(const std::string& model_path);

    /// Run a forward pass on a pre-built blob.
    /// @param blob 1x3xHxW float blob, normalized per `preprocessing.txt`.
    /// @return Vector of per-class probabilities, length == numClasses().
    std::vector<float> classify(const cv::Mat& blob);

    int numClasses() const;

private:
    Ort::Env                 env_;
    Ort::Session             session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::AllocatedStringPtr  input_name_;
    Ort::AllocatedStringPtr  output_name_;
    int                      num_classes_ = 0;
};

}  // namespace gr
