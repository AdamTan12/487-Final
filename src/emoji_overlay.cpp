#include "emoji_overlay.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>

// =============================================================================
//  Display-side implementation. The CV pipeline is on the other side of
//  gesture_types.h and never sees anything in this file.
//
//  Each gesture is a 72x72 BGRA Twemoji PNG in assets/emoji/. The PNGs are
//  loaded lazily on first use and cached for the rest of the run. Each frame
//  we:
//    1. Decide whether to draw (gesture != None && stable, with a 5-frame
//       fade-in on the first stable frame so it eases on instead of popping).
//    2. Resize the cached BGRA emoji to a size proportional to the hand bbox.
//    3. Compute a screen-clamped placement (centered above the box).
//    4. Alpha-blend the emoji onto the frame in place.
//
//  Path resolution is "assets/emoji/<name>.png" relative to the working dir,
//  which the build/run instructions specify as the project root. If the PNG
//  is missing we print one warning and continue with no overlay for that
//  gesture; we never throw.
// =============================================================================

namespace gd {

namespace {

// ---- asset table ----------------------------------------------------------

struct EmojiAsset {
    Gesture     gesture;
    const char* path;
};

constexpr std::array<EmojiAsset, 6> kAssets = {{
    {Gesture::Fist,    "assets/emoji/fist.png"},
    {Gesture::Peace,   "assets/emoji/peace.png"},
    {Gesture::Three,   "assets/emoji/three.png"},
    {Gesture::Four,    "assets/emoji/four.png"},
    {Gesture::Palm,    "assets/emoji/palm.png"},
    {Gesture::Ok,      "assets/emoji/ok.png"},
}};

// ---- sizing & fade tuning -------------------------------------------------

constexpr float kSizeFractionOfBbox = 0.55f;  ///< emoji square = this * bbox width
constexpr int   kMinEmojiSize       = 48;     ///< px, smallest emoji we will draw
constexpr int   kMaxEmojiSize       = 240;    ///< px, largest emoji we will draw
constexpr int   kGapAboveBbox       = 12;     ///< px between bbox top and emoji bottom
constexpr int   kFadeFrames         = 5;      ///< 0..5; controls ease-on

// ---- asset cache ----------------------------------------------------------

/// Lazy-loaded BGRA emoji table. Empty Mat = "tried to load, failed".
cv::Mat& assetFor(Gesture g) {
    static std::unordered_map<int, cv::Mat>  cache;
    static std::unordered_map<int, bool>     warned;
    const int key = static_cast<int>(g);

    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    cv::Mat& slot = cache[key];   // default-constructed empty Mat
    for (const auto& a : kAssets) {
        if (a.gesture != g) continue;
        cv::Mat img = cv::imread(a.path, cv::IMREAD_UNCHANGED);
        if (img.empty() || img.channels() != 4) {
            if (!warned[key]) {
                std::cerr << "emoji_overlay: failed to load BGRA from "
                          << a.path << " for gesture "
                          << gesture_name(g) << "\n";
                warned[key] = true;
            }
        } else {
            slot = img;  // keep as 72x72 BGRA; we resize per draw
        }
        break;
    }
    return slot;
}

// ---- alpha blending -------------------------------------------------------

/// Center the emoji square horizontally on the bbox and place it above the
/// bbox. If that would clip the top of the frame, push it down inside.
cv::Rect placeAboveBbox(const cv::Rect& bbox, int size, const cv::Size& frame) {
    int x = bbox.x + bbox.width / 2 - size / 2;
    int y = bbox.y - kGapAboveBbox - size;
    x = std::clamp(x, 0, std::max(0, frame.width  - size));
    y = std::clamp(y, 0, std::max(0, frame.height - size));
    return cv::Rect(x, y, size, size);
}

/// Blend a BGRA source onto a BGR frame, in place. `global_alpha` in [0, 1]
/// is multiplied with the PNG's per-pixel alpha (used for fade-in).
///
/// Linear blend per channel: out = src*a + dst*(1-a), with
///   a = src_alpha/255 * global_alpha
///
/// The dst ROI and src must be the same size; the caller does the resize and
/// the placement clamp.
void alphaBlend(cv::Mat& frame_roi, const cv::Mat& bgra, float global_alpha) {
    CV_Assert(frame_roi.type() == CV_8UC3);
    CV_Assert(bgra.type()      == CV_8UC4);
    CV_Assert(frame_roi.size() == bgra.size());

    const float clamped_global = std::clamp(global_alpha, 0.0f, 1.0f);

    for (int y = 0; y < bgra.rows; ++y) {
        const auto* src = bgra.ptr<cv::Vec4b>(y);
        auto*       dst = frame_roi.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgra.cols; ++x) {
            const float a = (static_cast<float>(src[x][3]) / 255.0f) * clamped_global;
            if (a <= 0.0f) continue;
            const float inv = 1.0f - a;
            for (int c = 0; c < 3; ++c) {
                dst[x][c] = static_cast<std::uint8_t>(
                    static_cast<float>(src[x][c]) * a +
                    static_cast<float>(dst[x][c]) * inv);
            }
        }
    }
}

// ---- fade state -----------------------------------------------------------

/// Tracks "how many consecutive frames have we been stable on this gesture?"
/// Reset whenever the gesture changes or stops being stable.
struct FadeState {
    Gesture last  = Gesture::None;
    int     count = 0;
};

FadeState& fadeState() {
    static FadeState s;
    return s;
}

}  // namespace

void draw_gesture(cv::Mat& frame, const GestureEvent& ev) {
    if (frame.empty() || frame.type() != CV_8UC3) {
        return;  // we draw onto BGR webcam frames only
    }

    FadeState& fade = fadeState();

    // Nothing worth showing -> reset the fade so the next gesture eases in.
    if (!ev.stable || ev.gesture == Gesture::None) {
        fade = {};
        return;
    }

    // Gesture changed (or first stable frame) -> restart fade-in.
    if (ev.gesture != fade.last) {
        fade.last  = ev.gesture;
        fade.count = 0;
    }
    fade.count = std::min(kFadeFrames, fade.count + 1);
    const float global_alpha = static_cast<float>(fade.count) /
                               static_cast<float>(kFadeFrames);

    // Bail if the asset failed to load.
    const cv::Mat& bgra = assetFor(ev.gesture);
    if (bgra.empty()) {
        return;
    }

    // Size proportional to bbox width, clamped to a sensible band.
    int size = static_cast<int>(static_cast<float>(ev.bbox.width) *
                                kSizeFractionOfBbox);
    size = std::clamp(size, kMinEmojiSize, kMaxEmojiSize);

    const cv::Rect place = placeAboveBbox(ev.bbox, size, frame.size());
    if (place.area() <= 0) {
        return;
    }

    cv::Mat resized;
    cv::resize(bgra, resized, place.size(), 0, 0, cv::INTER_AREA);

    cv::Mat roi = frame(place);
    alphaBlend(roi, resized, global_alpha);
}

}  // namespace gd
