#include "emoji_overlay.h"

#include <opencv2/imgproc.hpp>

namespace gd {

// =============================================================================
//  PLACEHOLDER IMPLEMENTATION  (teammate replaces this whole file)
//
//  For now this just prints the gesture name as text near the hand so the app
//  builds and runs end-to-end. Swap it for real emoji rendering:
//
//    1. Keep a Gesture -> cv::Mat (BGRA, with alpha) table loaded at startup
//       from PNG assets (cv::imread(path, cv::IMREAD_UNCHANGED)).
//    2. On draw_gesture, look up ev.gesture, then alpha-blend that PNG onto
//       `frame` at a spot derived from ev.bbox (e.g. centered above the box).
//    3. Use ev.stable to decide whether to show it (and optionally fade in).
//
//  NOTE: cv::putText cannot render real emoji glyphs (Hershey vector fonts
//  only) — that is exactly why the real implementation alpha-blends PNGs.
//
//  PARALLEL-DEV HARNESS: to build/test this without the CV pipeline, write a
//  tiny main that fills a GestureEvent by hand (e.g. keys 1-7 pick a gesture,
//  bbox = a fixed rect) and calls draw_gesture on a blank/webcam frame.
// =============================================================================

namespace {
const cv::Scalar kPlaceholderColor(0, 215, 255);  // amber
constexpr int    kFont      = cv::FONT_HERSHEY_SIMPLEX;
constexpr double kScale     = 1.2;
constexpr int    kThickness = 3;
}  // namespace

void draw_gesture(cv::Mat& frame, const GestureEvent& ev) {
    if (ev.gesture == Gesture::None || !ev.stable) {
        return;
    }
    const cv::Point org(ev.bbox.x, std::max(40, ev.bbox.y - 12));
    cv::putText(frame, std::string("[") + gesture_name(ev.gesture) + "]",
                org, kFont, kScale, kPlaceholderColor, kThickness);
}

}  // namespace gd
