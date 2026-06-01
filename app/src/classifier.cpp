#include "classifier.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gr {

namespace {

/// Numerically-stable softmax over a single row of logits.
std::vector<float> softmax(const float* logits, int n) {
    std::vector<float> out(n);
    const float max_v = *std::max_element(logits, logits + n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        out[i] = std::exp(logits[i] - max_v);
        sum   += out[i];
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (float& v : out) v *= inv;
    return out;
}

Ort::SessionOptions makeSessionOptions() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

}  // namespace

Classifier::Classifier(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "gr_classifier"),
      session_(env_, model_path.c_str(), makeSessionOptions()),
      input_name_(session_.GetInputNameAllocated(0, allocator_)),
      output_name_(session_.GetOutputNameAllocated(0, allocator_)) {}

std::vector<float> Classifier::classify(const cv::Mat& blob) {
    if (blob.empty() || blob.dims != 4) {
        throw std::runtime_error("Classifier::classify: expected 4-D NCHW blob");
    }
    const int64_t N = blob.size[0];
    const int64_t C = blob.size[1];
    const int64_t H = blob.size[2];
    const int64_t W = blob.size[3];
    const size_t  num_elems = static_cast<size_t>(N * C * H * W);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    const std::array<int64_t, 4> input_shape = {N, C, H, W};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(blob.ptr<float>()),
        num_elems,
        input_shape.data(),
        input_shape.size());

    const char* in_names[]  = {input_name_.get()};
    const char* out_names[] = {output_name_.get()};
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        in_names, &input_tensor, 1,
        out_names, 1);

    const float* logits = outputs[0].GetTensorData<float>();
    const auto   shape  = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.empty()) {
        throw std::runtime_error("Classifier::classify: empty output shape");
    }
    const int n = static_cast<int>(shape.back());
    num_classes_ = n;
    return softmax(logits, n);
}

int Classifier::numClasses() const {
    return num_classes_;
}

}  // namespace gr
