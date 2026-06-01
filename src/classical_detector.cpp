#include "classical_detector.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace gd {

namespace {

// --- skin segmentation (static HSV bootstrap) ------------------------------
constexpr int    kHueMin           = 0;
constexpr int    kHueMax           = 20;
constexpr int    kSatMin           = 30;
constexpr int    kSatMax           = 150;
constexpr int    kValMax           = 255;
constexpr int    kMorphKernelSize  = 7;
constexpr int    kBlurKernelSize   = 5;
constexpr double kMinContourArea   = 4000.0;

// --- face cascade ----------------------------------------------------------
constexpr int    kCascadeEveryN    = 3;       // run Haar on every Nth detect() call
constexpr double kFaceScaleFactor  = 1.2;
constexpr int    kFaceMinNeighbors = 4;
constexpr int    kFaceMinSizePx    = 80;
constexpr double kFacePadFraction  = 0.20;    // expand face box before zeroing in mask

// --- adaptive skin model ---------------------------------------------------
constexpr int    kHueBins              = 16;
constexpr int    kSatBins              = 16;
constexpr double kHistLearnRate        = 0.30;  // EMA weight on new samples
constexpr int    kMinTrainedFrames     = 3;     // wait this many face samples before trusting
constexpr int    kBackProjThreshold    = 50;    // 0..255; lower = more permissive
constexpr int    kBackProjBlur         = 5;     // odd, smooths the likelihood map
constexpr double kFacePatchInsetFrac   = 0.25;  // shrink face box this much per side for sampling
constexpr double kFacePatchVerticalShift = 0.05; // nudge patch up to skew toward forehead+cheeks

// --- shape scoring ---------------------------------------------------------
constexpr double kSolidityPower    = 1.5;
constexpr double kSolidityCap      = 0.97;
// ---------------------------------------------------------------------------

const std::array<const char*, 6> kCascadePaths = {
    "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/opt/homebrew/Cellar/opencv/4.13.0_10/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_default.xml",
};

/// Largest of `rects` by area, or an empty Rect if the vector is empty.
cv::Rect largestFace(const std::vector<cv::Rect>& rects) {
    cv::Rect best;
    int      best_area = 0;
    for (const auto& r : rects) {
        if (r.area() > best_area) {
            best_area = r.area();
            best      = r;
        }
    }
    return best;
}

/// Central patch of a face box — biased upward to favor forehead + cheeks
/// over jawline (which often picks up shadow / facial hair).
cv::Rect facePatch(const cv::Rect& face, const cv::Size& frame_size) {
    const int dx = static_cast<int>(face.width  * kFacePatchInsetFrac);
    const int dy = static_cast<int>(face.height * kFacePatchInsetFrac);
    const int vshift = static_cast<int>(face.height * kFacePatchVerticalShift);
    cv::Rect patch(face.x + dx,
                   face.y + dy - vshift,
                   face.width  - 2 * dx,
                   face.height - 2 * dy);
    patch &= cv::Rect(0, 0, frame_size.width, frame_size.height);
    return patch;
}

}  // namespace

ClassicalDetector::ClassicalDetector() {
    for (const auto* p : kCascadePaths) {
        if (face_cascade_.load(p)) {
            std::cerr << "ClassicalDetector: face cascade loaded from " << p << "\n";
            return;
        }
    }
    std::cerr << "ClassicalDetector: warning — face cascade not found; "
                 "face masking + adaptive skin disabled\n";
}

DetectionResult ClassicalDetector::detect(const cv::Mat& frame) {
    DetectionResult result;
    if (frame.empty()) {
        return result;
    }

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // ----- 1. Face cascade (every Nth frame, cached) -----------------------
    ++detect_calls_;
    const bool can_use_cascade = use_face_mask && !face_cascade_.empty();
    const bool ran_cascade     = can_use_cascade && (detect_calls_ % kCascadeEveryN == 0);
    if (ran_cascade) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        face_cascade_.detectMultiScale(gray, cached_faces_,
                                       kFaceScaleFactor,
                                       kFaceMinNeighbors,
                                       0,
                                       cv::Size(kFaceMinSizePx, kFaceMinSizePx));
    }
    result.faces = cached_faces_;

    // ----- 2. Update adaptive skin histogram from face patch ---------------
    if (use_adaptive_skin && ran_cascade && !cached_faces_.empty()) {
        const cv::Rect face  = largestFace(cached_faces_);
        const cv::Rect patch = facePatch(face, frame.size());
        if (patch.area() > 100) {
            cv::Mat new_hist;
            const cv::Mat face_hsv = hsv(patch);
            int           channels[] = {0, 1};
            int           hist_size[] = {kHueBins, kSatBins};
            float         h_range[] = {0.f, 180.f};
            float         s_range[] = {0.f, 256.f};
            const float*  ranges[]  = {h_range, s_range};
            cv::calcHist(&face_hsv, 1, channels, cv::Mat(),
                         new_hist, 2, hist_size, ranges);
            cv::normalize(new_hist, new_hist, 0.0, 255.0, cv::NORM_MINMAX);

            if (skin_hist_.empty()) {
                skin_hist_ = new_hist;
            } else {
                cv::addWeighted(skin_hist_, 1.0 - kHistLearnRate,
                                new_hist,    kHistLearnRate, 0.0, skin_hist_);
            }
            ++trained_frames_;
        }
    }
    result.trained_frames = trained_frames_;

    // ----- 3. Build skin mask: adaptive if trained, else static HSV --------
    cv::Mat mask;
    const bool adaptive_active = use_adaptive_skin &&
                                 !skin_hist_.empty() &&
                                 trained_frames_ >= kMinTrainedFrames;
    result.adaptive_active = adaptive_active;

    if (adaptive_active) {
        cv::Mat back_proj;
        int          channels[] = {0, 1};
        float        h_range[] = {0.f, 180.f};
        float        s_range[] = {0.f, 256.f};
        const float* ranges[]  = {h_range, s_range};
        cv::calcBackProject(&hsv, 1, channels, skin_hist_, back_proj, ranges);
        if (kBackProjBlur > 0) {
            cv::GaussianBlur(back_proj, back_proj,
                             cv::Size(kBackProjBlur, kBackProjBlur), 0);
        }
        cv::threshold(back_proj, mask, kBackProjThreshold, 255, cv::THRESH_BINARY);

        // Belt-and-suspenders: drop pixels that are clearly too dark to be
        // skin (the histogram only models hue+sat).
        cv::Mat value_ok;
        cv::inRange(hsv,
                    cv::Scalar(0, 0, v_min),
                    cv::Scalar(180, 255, 255),
                    value_ok);
        cv::bitwise_and(mask, value_ok, mask);
    } else {
        cv::inRange(hsv,
                    cv::Scalar(kHueMin, kSatMin, v_min),
                    cv::Scalar(kHueMax, kSatMax, kValMax),
                    mask);
    }

    // ----- 4. Cleanup ------------------------------------------------------
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(kMorphKernelSize, kMorphKernelSize));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    if (kBlurKernelSize > 0) {
        cv::GaussianBlur(mask, mask,
                         cv::Size(kBlurKernelSize, kBlurKernelSize), 0);
    }

    // ----- 5. Zero out face regions ----------------------------------------
    if (use_face_mask) {
        for (const auto& f : cached_faces_) {
            const int pad_x = static_cast<int>(f.width  * kFacePadFraction);
            const int pad_y = static_cast<int>(f.height * kFacePadFraction);
            cv::Rect padded(f.x - pad_x, f.y - pad_y,
                            f.width  + 2 * pad_x,
                            f.height + 2 * pad_y);
            padded &= cv::Rect(0, 0, mask.cols, mask.rows);
            if (padded.area() > 0) {
                mask(padded).setTo(0);
            }
        }
    }

    // ----- 6. Pick the best hand candidate ---------------------------------
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const std::vector<cv::Point>* best       = nullptr;
    double                        best_score = -1.0;
    for (const auto& c : contours) {
        const double area = cv::contourArea(c);
        if (area < kMinContourArea) continue;

        double score = area;
        if (use_shape_score) {
            std::vector<cv::Point> hull;
            cv::convexHull(c, hull);
            const double hull_area = cv::contourArea(hull);
            if (hull_area <= 0.0) continue;
            const double solidity = std::min(1.0, area / hull_area);
            if (solidity > kSolidityCap) continue;
            score = area * std::pow(1.0 - solidity, kSolidityPower);
        }

        if (score > best_score) {
            best_score = score;
            best       = &c;
        }
    }

    result.mask = mask;
    if (best == nullptr) {
        return result;
    }
    result.found   = true;
    result.contour = *best;
    result.bbox    = cv::boundingRect(*best);
    return result;
}

}  // namespace gd
