#include "camera.h"
#include "classical_classifier.h"
#include "classical_detector.h"
#include "emoji_overlay.h"
#include "gesture_types.h"
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
constexpr double      kDepthStep  = 1.0;
constexpr double      kDepthFloor = 1.0;
constexpr double      kDepthCeil  = 100.0;
constexpr int         kBpStep     = 5;
constexpr int         kBpFloor    = 0;
constexpr int         kBpCeil     = 255;
constexpr double      kSolStep    = 0.02;
constexpr double      kSolFloor   = 0.50;
constexpr double      kSolCeil    = 1.00;

/// Prints the keyboard shortcut reference to stdout on startup.
void printControls() {
    std::cout << "controls:\n"
              << "  ESC   quit\n"
              << "  d     toggle debug view (frame | mask)\n"
              << "  +/=   raise HSV v_min\n"
              << "  -/_   lower HSV v_min\n"
              << "  f     toggle face mask-out\n"
              << "  s     toggle hand-likeness shape scoring\n"
              << "  a     toggle adaptive skin model (learned from face)\n"
              << "  [/]   lower/raise convexity-defect depth threshold\n"
              << "  ,/.   lower/raise adaptive skin threshold (lower = picks up more hand)\n"
              << "  o/p   lower/raise fist-vs-palm solidity threshold\n";
}

}  // namespace

/// Application entry point.
/// Opens the default camera, runs the detect → classify → smooth → render
/// pipeline on every frame, and handles keyboard input to toggle detector
/// and classifier parameters at runtime.
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

        gd::Gesture raw_gesture  = gd::Gesture::None;
        int         finger_count = 0;
        if (det.found) {
            const auto cls = classifier.classify(det.contour, det.has_hole);
            raw_gesture  = cls.gesture;
            finger_count = cls.finger_count;
        }
        const gd::SmoothResult smoothed = smoother.update(raw_gesture);

        const int64  now = cv::getTickCount();
        const double dt  = static_cast<double>(now - prev_tick) / tick_freq;
        prev_tick = now;
        const double fps = (dt > 0.0) ? 1.0 / dt : 0.0;

        // our debug HUD
        renderer.draw(frame, det, smoothed.gesture, finger_count, fps,
                      detector.use_face_mask,
                      detector.use_shape_score,
                      detector.use_adaptive_skin);

        // hand off to the display side
        gd::GestureEvent ev;
        ev.gesture      = smoothed.gesture;
        ev.stable       = smoothed.stable;
        ev.bbox         = det.bbox;
        ev.finger_count = finger_count;
        gd::draw_gesture(frame, ev);

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
        } else if (key == ']' || key == '}') {
            classifier.defect_depth_min = std::min(
                kDepthCeil, classifier.defect_depth_min + kDepthStep);
            std::cout << "defect_depth_min = "
                      << classifier.defect_depth_min << "\n";
        } else if (key == '[' || key == '{') {
            classifier.defect_depth_min = std::max(
                kDepthFloor, classifier.defect_depth_min - kDepthStep);
            std::cout << "defect_depth_min = "
                      << classifier.defect_depth_min << "\n";
        } else if (key == '.' || key == '>') {
            detector.backproj_threshold = std::min(
                kBpCeil, detector.backproj_threshold + kBpStep);
            std::cout << "backproj_threshold = "
                      << detector.backproj_threshold << "\n";
        } else if (key == ',' || key == '<') {
            detector.backproj_threshold = std::max(
                kBpFloor, detector.backproj_threshold - kBpStep);
            std::cout << "backproj_threshold = "
                      << detector.backproj_threshold << "\n";
        } else if (key == 'p' || key == 'P') {
            classifier.open_solidity_max = std::min(
                kSolCeil, classifier.open_solidity_max + kSolStep);
            std::cout << "open_solidity_max = "
                      << classifier.open_solidity_max << "\n";
        } else if (key == 'o' || key == 'O') {
            classifier.open_solidity_max = std::max(
                kSolFloor, classifier.open_solidity_max - kSolStep);
            std::cout << "open_solidity_max = "
                      << classifier.open_solidity_max << "\n";
        }
    }

    cv::destroyAllWindows();
    return 0;
}
