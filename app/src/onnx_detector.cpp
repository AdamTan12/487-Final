#include "onnx_detector.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace gr {

namespace {

const cv::Scalar kDetectorPad(144, 144, 144);

/// Letterbox resize: scale-to-fit + center-pad. Returns the canvas and
/// the geometry needed to map detections back to original frame coords.
struct LetterboxInfo {
    cv::Mat canvas;
    double  scale;
    int     left;
    int     top;
};

LetterboxInfo letterboxFrame(const cv::Mat& src, int size) {
    const int h = src.rows;
    const int w = src.cols;
    const double scale = static_cast<double>(size) / std::max(h, w);
    const int new_w = std::max(1, static_cast<int>(std::round(w * scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(h * scale)));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(size, size, src.type(), kDetectorPad);
    const int top  = (size - new_h) / 2;
    const int left = (size - new_w) / 2;
    resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));
    return {canvas, scale, left, top};
}

Ort::SessionOptions makeSessionOptions() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

}  // namespace

OnnxDetector::OnnxDetector(const std::string& model_path,
                           const cv::Size&    input_size,
                           float              conf_threshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "gr_detector"),
      session_(env_, model_path.c_str(), makeSessionOptions()),
      input_name_(session_.GetInputNameAllocated(0, allocator_)),
      output_name_(session_.GetOutputNameAllocated(0, allocator_)),
      input_size_(input_size),
      conf_threshold_(conf_threshold) {}

std::optional<cv::Rect> OnnxDetector::detect(const cv::Mat& frame) {
    if (frame.empty()) {
        return std::nullopt;
    }

    // 1. Letterbox to the model's expected square input.
    const auto lb = letterboxFrame(frame, input_size_.width);

    // 2. Build the NCHW float blob -- normalize to [0,1] and swap to RGB.
    //    We keep cv::dnn::blobFromImage for the *preprocessing* (a pure
    //    image transform with no neural-net inference) -- it's a clean
    //    helper for HWC->NCHW packing. Inference itself runs through
    //    onnxruntime.
    cv::Mat blob = cv::dnn::blobFromImage(
        lb.canvas,
        1.0 / 255.0,
        input_size_,
        cv::Scalar(0, 0, 0),
        /*swapRB=*/true,
        /*crop=*/false,
        CV_32F);

    // 3. Forward through onnxruntime.
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> shape = {1, 3, input_size_.height, input_size_.width};
    const size_t num_elems = static_cast<size_t>(shape[0] * shape[1] * shape[2] * shape[3]);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(blob.ptr<float>()),
        num_elems,
        shape.data(),
        shape.size());

    const char* in_names[]  = {input_name_.get()};
    const char* out_names[] = {output_name_.get()};
    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        in_names, &input_tensor, 1,
        out_names, 1);

    const float* data    = outputs[0].GetTensorData<float>();
    const auto   out_dim = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (out_dim.size() != 3 || out_dim[2] < 6) {
        throw std::runtime_error("OnnxDetector: unexpected output shape");
    }
    const int num_dets = static_cast<int>(out_dim[1]);

    // 4. Pick the top-1 confident detection. YOLOv10's NMS-free head
    //    sorts detections by score, so #0 is usually the answer, but
    //    we scan all just to be safe.
    int   best_idx   = -1;
    float best_score = conf_threshold_;
    for (int i = 0; i < num_dets; ++i) {
        const float score = data[i * 6 + 4];
        if (score > best_score) {
            best_score = score;
            best_idx   = i;
        }
    }

    // Set GR_DEBUG_DETECTOR=1 to dump the top-3 raw detections per frame.
    static const bool dbg = std::getenv("GR_DEBUG_DETECTOR") != nullptr;
    if (dbg) {
        std::cerr << "det: ";
        for (int k = 0; k < std::min(3, num_dets); ++k) {
            const float* r = data + k * 6;
            std::cerr << "[" << r[0] << "," << r[1] << "," << r[2] << ","
                      << r[3] << " s=" << r[4] << " c=" << r[5] << "] ";
        }
        std::cerr << "best_idx=" << best_idx << " best_score=" << best_score << "\n";
    }

    if (best_idx < 0) {
        return std::nullopt;
    }

    // 5. Undo the letterbox to map back to original frame coordinates.
    const float* row = data + best_idx * 6;
    const auto unmap_x = [&](float v) {
        return static_cast<int>(std::round((v - static_cast<float>(lb.left)) / lb.scale));
    };
    const auto unmap_y = [&](float v) {
        return static_cast<int>(std::round((v - static_cast<float>(lb.top)) / lb.scale));
    };

    int x1 = std::max(0,           unmap_x(row[0]));
    int y1 = std::max(0,           unmap_y(row[1]));
    int x2 = std::min(frame.cols,  unmap_x(row[2]));
    int y2 = std::min(frame.rows,  unmap_y(row[3]));
    if (x2 <= x1 || y2 <= y1) {
        return std::nullopt;
    }
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

}  // namespace gr
