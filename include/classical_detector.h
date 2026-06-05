#pragma once

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <vector>

namespace gd {

/// All outputs produced by ClassicalDetector::detect() for a single frame.
struct DetectionResult {
    bool                   found = false;          ///< True if a plausible hand contour was found.
    std::vector<cv::Point> contour;                ///< Outer contour of the best hand candidate.
    cv::Rect               bbox;                   ///< Bounding rectangle of `contour`.
    cv::Mat                mask;                   ///< Final binary skin mask (before contour selection).
    std::vector<cv::Rect>  faces;                  ///< Face detections from this frame (for debug overlay).
    bool                   adaptive_active = false;///< True when the back-projection model is trained and in use.
    int                    trained_frames  = 0;    ///< Number of face-patch samples incorporated into the skin histogram.
    bool                   has_hole = false;       ///< True when the hand contour encloses a significant gap (e.g. the OK loop).
};

/// HSV-based hand detector with three optional enhancement modes:
///
///   Face mask (f key): erases the face region from the skin mask so the face
///     is never mistaken for a hand.
///
///   Shape scoring (s key): ranks candidate contours by
///     area × (1 − solidity)^k rather than plain area, favouring hand-shaped
///     regions over compact blobs.
///
///   Adaptive skin model (a key): builds a running H-S histogram from face
///     patches and uses back-projection as the skin mask once enough samples
///     have been collected, replacing the fixed HSV range with a
///     person-specific colour model.
///
/// The Haar face cascade runs only every few frames; the last result is reused
/// in between to keep the per-frame cost low.
class ClassicalDetector {
public:
    /// Loads the Haar frontal-face cascade from a list of known install paths.
    /// If no cascade file is found, face masking and adaptive skin are silently
    /// disabled and a warning is printed to stderr.
    ClassicalDetector();

    /// Runs the full detection pipeline on `frame` (BGR, 8-bit) and returns
    /// the best hand candidate along with diagnostic fields.
    DetectionResult detect(const cv::Mat& frame);

    /// Minimum HSV value channel threshold for skin pixels. Raise to reject
    /// dark shadows. Tunable at runtime with the +/= and -/_ keys.
    int  v_min             = 60;

    /// When true, detected face regions are blacked out of the skin mask before
    /// contour search. Toggle with the f key.
    bool use_face_mask     = true;

    /// When true, contour candidates are ranked by a shape score that favours
    /// hand-like solidity over compact blobs. Toggle with the s key.
    bool use_shape_score   = true;

    /// When true, the detector learns skin colour from face patches and
    /// switches to back-projection once enough samples are collected. Toggle
    /// with the a key.
    bool use_adaptive_skin = true;

    /// Back-projection likelihood threshold (0–255). Lower values include more
    /// of the frame as skin, higher values are more conservative. Tunable at
    /// runtime with the , and . keys.
    int  backproj_threshold = 35;

private:
    cv::CascadeClassifier face_cascade_;

    std::vector<cv::Rect> cached_faces_;  ///< Face detections from the last cascade run.
    int                   detect_calls_ = 0;

    cv::Mat skin_hist_;         ///< Running H-S histogram built from face patches via EMA.
    int     trained_frames_ = 0;
};

}  // namespace gd
