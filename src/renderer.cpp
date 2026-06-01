#include "renderer.h"

#include <opencv2/imgproc.hpp>

#include <sstream>
#include <vector>

namespace gd {

namespace {

const cv::Scalar kBboxColor    (0, 255, 0);     // green
const cv::Scalar kContourColor (0, 140, 255);   // orange
const cv::Scalar kTextColor    (255, 255, 255); // white
const cv::Scalar kPanelColor   (40, 40, 40);    // dark grey

constexpr int    kFont          = cv::FONT_HERSHEY_SIMPLEX;
constexpr double kLabelScale    = 0.8;
constexpr double kPanelScale    = 0.55;
constexpr int    kLabelThick    = 2;
constexpr int    kPanelThick    = 1;
constexpr int    kPanelPadX     = 10;
constexpr int    kPanelPadY     = 10;
constexpr int    kPanelLineGap  = 22;
constexpr int    kPanelWidth    = 300;
constexpr int    kPanelHeight   = 175;

const cv::Scalar kFaceBoxColor (0, 0, 255);    // red, debug view only

std::string fmt1(double v) {
    std::ostringstream oss;
    oss.precision(1);
    oss << std::fixed << v;
    return oss.str();
}

}  // namespace

void Renderer::draw(cv::Mat&               frame,
                    const DetectionResult& det,
                    Gesture                smoothed_gesture,
                    int                    finger_count,
                    double                 fps,
                    bool                   face_mask_on,
                    bool                   shape_score_on,
                    bool                   adaptive_on) {
    const std::string smoothed_label = gesture_name(smoothed_gesture);
    if (det.found) {
        cv::rectangle(frame, det.bbox, kBboxColor, 2);
        const std::vector<std::vector<cv::Point>> contours{det.contour};
        cv::drawContours(frame, contours, 0, kContourColor, 2);

        const cv::Point label_org(det.bbox.x,
                                  std::max(20, det.bbox.y - 8));
        cv::putText(frame, smoothed_label, label_org,
                    kFont, kLabelScale, kBboxColor, kLabelThick);
    }

    // status panel, top-left
    cv::Rect panel(kPanelPadX, kPanelPadY, kPanelWidth, kPanelHeight);
    panel &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (panel.area() > 0) {
        cv::Mat roi = frame(panel);
        cv::Mat overlay(roi.size(), roi.type(), kPanelColor);
        cv::addWeighted(overlay, 0.55, roi, 0.45, 0.0, roi);
    }

    const int x = kPanelPadX + 8;
    int       y = kPanelPadY + kPanelLineGap;
    cv::putText(frame, "label: " + smoothed_label,
                cv::Point(x, y), kFont, kPanelScale, kTextColor, kPanelThick);
    y += kPanelLineGap;
    cv::putText(frame, "fingers: " + std::to_string(finger_count),
                cv::Point(x, y), kFont, kPanelScale, kTextColor, kPanelThick);
    y += kPanelLineGap;
    cv::putText(frame, "fps: " + fmt1(fps),
                cv::Point(x, y), kFont, kPanelScale, kTextColor, kPanelThick);
    y += kPanelLineGap;
    cv::putText(frame,
                std::string("face[f]:") + (face_mask_on   ? "on " : "off") +
                std::string("  shape[s]:") + (shape_score_on ? "on" : "off"),
                cv::Point(x, y), kFont, kPanelScale * 0.85, kTextColor, kPanelThick);
    y += kPanelLineGap;
    std::string adapt_status;
    if (!adaptive_on) {
        adapt_status = "off";
    } else if (det.adaptive_active) {
        adapt_status = "on (trained " + std::to_string(det.trained_frames) + ")";
    } else {
        adapt_status = "on (training " + std::to_string(det.trained_frames) + ")";
    }
    cv::putText(frame, "adapt[a]: " + adapt_status,
                cv::Point(x, y), kFont, kPanelScale * 0.85, kTextColor, kPanelThick);
}

cv::Mat Renderer::debugView(const cv::Mat& frame, const DetectionResult& det) {
    if (det.mask.empty()) {
        return frame.clone();
    }
    cv::Mat mask_bgr;
    cv::cvtColor(det.mask, mask_bgr, cv::COLOR_GRAY2BGR);
    if (mask_bgr.size() != frame.size()) {
        cv::resize(mask_bgr, mask_bgr, frame.size());
    }
    // red boxes on the left show what's getting cut out of the mask.
    cv::Mat left = frame.clone();
    for (const auto& f : det.faces) {
        cv::rectangle(left, f, kFaceBoxColor, 2);
    }
    cv::Mat out;
    cv::hconcat(left, mask_bgr, out);
    return out;
}

}  // namespace gd
