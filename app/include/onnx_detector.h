#pragma once

/// @file onnx_detector.h
/// @brief ONNX-based hand detector (HaGRID's pretrained YOLOv10n).

#include "i_hand_detector.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>

namespace gr {

/// YOLOv10n hand detector loaded through Microsoft's onnxruntime.
///
/// The model is HaGRID's pretrained `YOLOv10n_hands` (or compatible).
/// Input is a 640x640 letterboxed RGB blob; output is decoded as a
/// list of (xyxy, score, class) and the top-1 box is returned in
/// original frame coordinates. YOLOv10 is NMS-free by design, so no
/// post-processing NMS pass is needed.
class OnnxDetector : public IHandDetector {
public:
    OnnxDetector(const std::string& model_path,
                 const cv::Size& input_size  = cv::Size(640, 640),
                 float conf_threshold        = 0.50f);

    std::optional<cv::Rect> detect(const cv::Mat& frame) override;

private:
    Ort::Env                 env_;
    Ort::Session             session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::AllocatedStringPtr  input_name_;
    Ort::AllocatedStringPtr  output_name_;
    cv::Size                 input_size_;
    float                    conf_threshold_;
};

}  // namespace gr
