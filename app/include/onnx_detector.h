#pragma once

/// @file onnx_detector.h
/// @brief ONNX-based hand detector (HaGRID's pretrained YOLOv10n).

#include "i_hand_detector.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>

namespace gr {

/// YOLOv10-style ONNX detector loaded through cv::dnn.
///
/// Loaded model is expected to be HaGRID's pretrained `YOLOv10n` hand
/// detector (or compatible). Input is a 640x640 letterboxed RGB blob;
/// output is decoded with NMS and the top-1 box returned.
class OnnxDetector : public IHandDetector {
public:
    OnnxDetector(const std::string& model_path,
                 const cv::Size& input_size  = cv::Size(640, 640),
                 float conf_threshold        = 0.25f,
                 float nms_threshold         = 0.45f);

    std::optional<cv::Rect> detect(const cv::Mat& frame) override;

private:
    cv::dnn::Net net_;
    cv::Size     input_size_;
    float        conf_threshold_;
    float        nms_threshold_;
};

}  // namespace gr
