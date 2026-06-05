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

/// Returns a reference to the cached BGRA image for gesture `g`, loading it
/// from disk on first access. Returns an empty Mat if the PNG is missing or
/// not 4-channel; a warning is printed once per missing asset.
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

/// Computes the placement rect for a square emoji of side `size` centred
/// horizontally over `bbox` and positioned just above it. If the natural
/// position would clip outside the frame, the rect is clamped inward.
cv::Rect placeAboveBbox(const cv::Rect& bbox, int size, const cv::Size& frame) {
    int x = bbox.x + bbox.width / 2 - size / 2;
    int y = bbox.y - kGapAboveBbox - size;
    x = std::clamp(x, 0, std::max(0, frame.width  - size));
    y = std::clamp(y, 0, std::max(0, frame.height - size));
    return cv::Rect(x, y, size, size);
}

/// Alpha-blends a BGRA source image onto a BGR frame ROI in place.
/// `global_alpha` (0–1) is multiplied with each pixel's own alpha channel,
/// allowing the caller to control overall opacity independently of the PNG
/// transparency (used for the fade-in animation).
/// Per-pixel blend: out = src * a + dst * (1 − a),  where a = src_alpha/255 * global_alpha.
/// Caller must ensure frame_roi and bgra are the same size.
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

/// Tracks consecutive stable frames for the current gesture to drive fade-in.
/// Resets whenever the gesture changes or stability is lost.
struct FadeState {
    Gesture last  = Gesture::None;
    int     count = 0;
};

/// Returns a reference to the process-lifetime FadeState singleton.
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
