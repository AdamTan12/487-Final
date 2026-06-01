// =============================================================================
//  Standalone harness for the emoji overlay.
//
//  Drives gd::draw_gesture without the CV pipeline so the display side can be
//  built and visually verified independently. Opens the webcam, paints a
//  fixed bbox roughly where a hand might appear, and lets the operator pick
//  the displayed gesture with the number keys.
//
//  When the real CV pipeline is wired in (in src/main.cpp), delete this file
//  and the matching emoji_demo target in CMakeLists.txt -- no other change.
//
//  Controls:
//    1  Fist     2  Like     3  Dislike   4  Peace
//    5  Three    6  Four     7  Palm
//    0  clear (gesture = None, stable = false)
//    ESC quit
// =============================================================================

#include "emoji_overlay.h"
#include "gesture_types.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>

namespace {

constexpr const char* kWindow = "emoji_demo";
constexpr int         kEsc    = 27;

gd::Gesture gestureFromKey(int key) {
    switch (key) {
        case '1': return gd::Gesture::Fist;
        case '2': return gd::Gesture::Like;
        case '3': return gd::Gesture::Dislike;
        case '4': return gd::Gesture::Peace;
        case '5': return gd::Gesture::Three;
        case '6': return gd::Gesture::Four;
        case '7': return gd::Gesture::Palm;
        default:  return gd::Gesture::None;
    }
}

}  // namespace

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "emoji_demo: failed to open webcam\n";
        return 1;
    }

    cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);
    std::cout << "keys: 1 fist  2 like  3 dislike  4 peace  5 three  6 four  7 palm"
              << "  0 clear  ESC quit\n";

    gd::GestureEvent ev;  // starts as None, stable=false

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            continue;
        }

        // Fixed bbox in the lower middle of the frame -- pretend "the hand
        // is here." Width = 1/4 of frame width, square-ish.
        const int bw = frame.cols / 4;
        const int bh = bw;
        const int bx = (frame.cols - bw) / 2;
        const int by = frame.rows - bh - frame.rows / 8;
        ev.bbox = cv::Rect(bx, by, bw, bh);

        cv::rectangle(frame, ev.bbox, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "[mock bbox]",
                    cv::Point(ev.bbox.x, ev.bbox.y + ev.bbox.height + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1);

        gd::draw_gesture(frame, ev);

        cv::imshow(kWindow, frame);
        const int k = cv::waitKey(1);
        if (k == kEsc) {
            break;
        }
        if (k == '0') {
            ev = {};
            std::cout << "cleared\n";
        } else if (k >= '1' && k <= '7') {
            ev.gesture      = gestureFromKey(k);
            ev.stable       = true;
            ev.finger_count = 0;
            std::cout << "gesture = " << gd::gesture_name(ev.gesture) << "\n";
        }
    }

    return 0;
}
