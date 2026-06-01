#include "camera.h"
#include "classical_classifier.h"
#include "classical_detector.h"
#include "renderer.h"
#include "smoother.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

namespace {

constexpr const char* kWindowName = "gesture_demo";
constexpr int         kEscKey     = 27;
constexpr int         kVMinStep   = 5;
constexpr int         kVMinFloor  = 0;
constexpr int         kVMinCeil   = 255;

void printControls() {
    std::cout << "controls:\n"
              << "  ESC   quit\n"
              << "  d     toggle debug view (frame | mask)\n"
              << "  +/=   raise HSV v_min\n"
              << "  -/_   lower HSV v_min\n"
              << "  f     toggle face mask-out\n"
              << "  s     toggle hand-likeness shape scoring\n"
              << "  a     toggle adaptive skin model (learned from face)\n";
}

}  // namespace

int main() {
    gd::Camera camera;
    if (!camera.isOpen()) {
        return 1;
    }

    gd::ClassicalDetector   detector;
    gd::ClassicalClassifier classifier;
    gd::Smoother            smoother;
    gd::Renderer            renderer;

    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
    printControls();

    bool         debug_view = false;
    cv::Mat      frame;
    int64        prev_tick  = cv::getTickCount();
    const double tick_freq  = cv::getTickFrequency();

    while (true) {
        if (!camera.readFrame(frame)) {
            std::cerr << "main: dropped frame, retrying\n";
            continue;
        }

        const gd::DetectionResult det = detector.detect(frame);

        std::string raw_label    = "none";
        int         finger_count = 0;
        if (det.found) {
            const auto cls = classifier.classify(det.contour);
            raw_label    = cls.label;
            finger_count = cls.finger_count;
        }
        const std::string smoothed = smoother.update(raw_label);

        const int64  now = cv::getTickCount();
        const double dt  = static_cast<double>(now - prev_tick) / tick_freq;
        prev_tick = now;
        const double fps = (dt > 0.0) ? 1.0 / dt : 0.0;

        renderer.draw(frame, det, smoothed, finger_count, fps,
                      detector.use_face_mask,
                      detector.use_shape_score,
                      detector.use_adaptive_skin);

        cv::imshow(kWindowName,
                   debug_view ? renderer.debugView(frame, det) : frame);

        const int key = cv::waitKey(1);
        if (key == kEscKey) {
            break;
        } else if (key == 'd' || key == 'D') {
            debug_view = !debug_view;
            std::cout << "debug_view = " << (debug_view ? "on" : "off") << "\n";
        } else if (key == '+' || key == '=') {
            detector.v_min = std::min(kVMinCeil, detector.v_min + kVMinStep);
            std::cout << "v_min = " << detector.v_min << "\n";
        } else if (key == '-' || key == '_') {
            detector.v_min = std::max(kVMinFloor, detector.v_min - kVMinStep);
            std::cout << "v_min = " << detector.v_min << "\n";
        } else if (key == 'f' || key == 'F') {
            detector.use_face_mask = !detector.use_face_mask;
            std::cout << "face_mask = "
                      << (detector.use_face_mask ? "on" : "off") << "\n";
        } else if (key == 's' || key == 'S') {
            detector.use_shape_score = !detector.use_shape_score;
            std::cout << "shape_score = "
                      << (detector.use_shape_score ? "on" : "off") << "\n";
        } else if (key == 'a' || key == 'A') {
            detector.use_adaptive_skin = !detector.use_adaptive_skin;
            std::cout << "adaptive_skin = "
                      << (detector.use_adaptive_skin ? "on" : "off") << "\n";
        }
    }

    cv::destroyAllWindows();
    return 0;
}
