#pragma once

/// @file config.h
/// @brief Runtime configuration for the live webcam pipeline.

#include <opencv2/core.hpp>

#include <string>

namespace gr {

/// All settings the pipeline needs. Populated from CLI args + defaults.
/// Preprocessing fields (input_size / mean / std / scale) MUST match the
/// HaGRID training pipeline byte-for-byte. See models/preprocessing.txt.
struct AppConfig {
    bool         debug               = false;

    std::string  model_path          = "models/gesture_classifier.onnx";
    std::string  detector_model_path = "models/hand_detector.onnx";
    std::string  class_names_path    = "models/class_names.txt";

    int          camera_index        = 0;

    // HaGRID ResNet18 preprocessing -- see models/preprocessing.txt.
    cv::Size     input_size          = cv::Size(224, 224);
    cv::Scalar   mean                = cv::Scalar(0.54,  0.499, 0.474);
    cv::Scalar   std                 = cv::Scalar(0.234, 0.235, 0.231);
    float        scale               = 1.0f / 255.0f;
    cv::Scalar   pad_color           = cv::Scalar(144, 144, 144);
};

/// Parse argv into an AppConfig. On `--help` or parse error, prints usage
/// to stderr and exits the process.
AppConfig parseArgs(int argc, char** argv);

}  // namespace gr
