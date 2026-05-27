#pragma once

/// @file config.h
/// @brief Runtime configuration shared across pipeline modules.

#include <opencv2/core.hpp>

#include <string>

namespace gr {

enum class DetectorType {
    Classical,  ///< HSV + contours + morphology (primary)
    Onnx,       ///< YOLOv10n (HaGRID pretrained) via cv::dnn (fallback)
};

enum class AppMode {
    Live,
    Eval,
    Bench,
};

/// All settings the pipeline needs. Populated from CLI args + defaults.
/// Preprocessing fields (input_size / mean / std / scale) MUST match the
/// Python training pipeline byte-for-byte. See models/preprocessing.txt.
struct AppConfig {
    AppMode      mode               = AppMode::Live;
    DetectorType detector            = DetectorType::Classical;
    bool         debug               = false;

    std::string  model_path          = "models/gesture_classifier.onnx";
    std::string  detector_model_path = "models/hand_detector.onnx";
    std::string  class_names_path    = "models/class_names.txt";

    int          camera_index        = 0;

    cv::Size     input_size          = cv::Size(224, 224);
    cv::Scalar   mean                = cv::Scalar(0.485, 0.456, 0.406);
    cv::Scalar   std                 = cv::Scalar(0.229, 0.224, 0.225);
    float        scale               = 1.0f / 255.0f;

    int          smoothing_window    = 5;

    // Mode-specific
    std::string  eval_dir            = "tests/test_images";
    std::string  eval_output_csv     = "tests/results.csv";
};

/// Parse argv into an AppConfig. On `--help` or parse error, prints usage
/// to stderr and exits the process.
AppConfig parseArgs(int argc, char** argv);

}  // namespace gr
