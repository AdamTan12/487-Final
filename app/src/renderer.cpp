#include "renderer.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <sstream>
#include <utility>

namespace gr {

namespace {

constexpr int        kFont       = cv::FONT_HERSHEY_SIMPLEX;
constexpr double     kFontScale  = 0.6;
constexpr int        kThickness  = 2;
const cv::Scalar     kBoxColor   = cv::Scalar(0, 255, 0);    // green
const cv::Scalar     kLabelColor = cv::Scalar(0, 255, 0);
const cv::Scalar     kFpsColor   = cv::Scalar(255, 255, 255);

}  // namespace

Renderer::Renderer(std::vector<std::string> class_names)
    : class_names_(std::move(class_names)) {}

void Renderer::draw(cv::Mat&                       frame,
                    const std::optional<cv::Rect>& bbox,
                    int                            class_id,
                    float                          confidence,
                    const std::vector<float>&      /*all_scores*/,
                    double                         fps) {
    if (bbox) {
        cv::rectangle(frame, *bbox, kBoxColor, kThickness);

        if (class_id >= 0 && class_id < static_cast<int>(class_names_.size())) {
            std::ostringstream label;
            label << class_names_[class_id] << " " << static_cast<int>(confidence * 100) << "%";
            const int y = std::max(20, bbox->y - 8);
            cv::putText(frame, label.str(), cv::Point(bbox->x, y),
                        kFont, kFontScale, kLabelColor, kThickness);
        }
    }

    std::ostringstream fps_text;
    fps_text << "fps: " << static_cast<int>(fps + 0.5);
    cv::putText(frame, fps_text.str(), cv::Point(10, 25),
                kFont, kFontScale, kFpsColor, kThickness);
}

}  // namespace gr
