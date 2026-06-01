#include "classical_detector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace gd {

namespace {

// --- tunable thresholds -----------------------------------------------------
constexpr int    kHueMin           = 0;
constexpr int    kHueMax           = 20;
constexpr int    kSatMin           = 30;
constexpr int    kSatMax           = 150;
constexpr int    kValMax           = 255;
constexpr int    kMorphKernelSize  = 7;     // ellipse, odd
constexpr int    kBlurKernelSize   = 5;     // odd, set to 0 to disable
constexpr double kMinContourArea   = 4000.0;
// ---------------------------------------------------------------------------

}  // namespace

DetectionResult ClassicalDetector::detect(const cv::Mat& frame) {
    DetectionResult result;
    if (frame.empty()) {
        return result;
    }

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv,
                cv::Scalar(kHueMin, kSatMin, v_min),
                cv::Scalar(kHueMax, kSatMax, kValMax),
                mask);

    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(kMorphKernelSize, kMorphKernelSize));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    if (kBlurKernelSize > 0) {
        cv::GaussianBlur(mask, mask,
                         cv::Size(kBlurKernelSize, kBlurKernelSize), 0);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    auto largest = std::max_element(
        contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });

    result.mask = mask;
    if (largest == contours.end() || cv::contourArea(*largest) < kMinContourArea) {
        return result;
    }

    result.found   = true;
    result.contour = *largest;
    result.bbox    = cv::boundingRect(*largest);
    return result;
}

}  // namespace gd
