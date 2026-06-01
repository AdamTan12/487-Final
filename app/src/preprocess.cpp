#include "preprocess.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace gr {

namespace {

/// Resize `src` so its longest side equals `size`, then center-pad to
/// `size x size` with `pad_color`. Returns the canvas, no metadata --
/// the classifier doesn't need to map back to original coords.
cv::Mat letterbox(const cv::Mat& src, int size, const cv::Scalar& pad_color) {
    const int h = src.rows;
    const int w = src.cols;
    const double scale = static_cast<double>(size) / std::max(h, w);
    const int new_w = std::max(1, static_cast<int>(std::round(w * scale)));
    const int new_h = std::max(1, static_cast<int>(std::round(h * scale)));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(size, size, src.type(), pad_color);
    const int top  = (size - new_h) / 2;
    const int left = (size - new_w) / 2;
    resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));
    return canvas;
}

}  // namespace

cv::Mat prepareBlob(const cv::Mat&   frame,
                    const cv::Rect&  bbox,
                    const AppConfig& cfg) {
    // Clamp bbox to frame so the crop is always valid.
    const cv::Rect safe = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    cv::Mat crop = (safe.area() > 0) ? frame(safe).clone() : frame.clone();

    // Letterbox in BGR with the configured gray pad color.
    cv::Mat square = letterbox(crop, cfg.input_size.width, cfg.pad_color);

    // cv::dnn::blobFromImage handles scale, mean-subtract, swapRB (BGR->RGB),
    // and HWC->NCHW packing in one call. Std-divide isn't in its API, so we
    // bake it into the mean/scale by hand: result = (x*scale - mean) / std.
    // We do that by first running blobFromImage with scale and mean=0,
    // then subtracting mean / dividing std manually as a separate step.
    cv::Mat blob = cv::dnn::blobFromImage(
        square,
        cfg.scale,                    // 1/255
        cfg.input_size,
        cv::Scalar(0, 0, 0),
        /*swapRB=*/true,              // model trained on RGB
        /*crop=*/false,
        CV_32F);

    // blob layout is NCHW: shape {1, 3, H, W}. Apply per-channel
    // (x - mean) / std in place.
    const int C = blob.size[1];
    const int H = blob.size[2];
    const int W = blob.size[3];
    const int plane = H * W;
    for (int c = 0; c < C; ++c) {
        const float mean_c = static_cast<float>(cfg.mean[c]);
        const float std_c  = static_cast<float>(cfg.std[c]);
        float* p = blob.ptr<float>(0, c);
        for (int i = 0; i < plane; ++i) {
            p[i] = (p[i] - mean_c) / std_c;
        }
    }

    return blob;
}

}  // namespace gr
