#pragma once

/// @file preprocess.h
/// @brief Build the classifier input blob from a detected hand region.

#include "config.h"

#include <opencv2/core.hpp>

namespace gr {

/// Crop the detected bbox out of `frame`, pad, resize, normalize, and
/// pack into a cv::dnn-compatible blob.
///
/// CONTRACT — this MUST match the Python training pipeline exactly:
///   * input_size: cfg.input_size
///   * channel order: BGR (no swap)
///   * pixel scale: cfg.scale (typically 1/255)
///   * normalization: (x - cfg.mean) / cfg.std, per channel
///
/// Any drift between Python preprocessing and this function will silently
/// degrade accuracy. Keep `models/preprocessing.txt` as the source of
/// truth and update both sides in lockstep.
///
/// @return 1x3xHxW float blob suitable for cv::dnn::Net::setInput.
cv::Mat prepareBlob(const cv::Mat&    frame,
                    const cv::Rect&   bbox,
                    const AppConfig&  cfg);

}  // namespace gr
