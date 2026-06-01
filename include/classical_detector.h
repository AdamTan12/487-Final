#pragma once

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <vector>

namespace gd {

struct DetectionResult {
    bool                   found = false;
    std::vector<cv::Point> contour;
    cv::Rect               bbox;
    cv::Mat                mask;            // final binary mask
    std::vector<cv::Rect>  faces;           // this frame's face hits (for debug)
    bool                   adaptive_active = false;  // back-projection in use?
    int                    trained_frames  = 0;      // face samples seen so far
    bool                   has_hole = false;         // enclosed gap, e.g. OK loop
};

// HSV skin detector with three toggles:
//   f - cut out detected faces before finding contours
//   s - rank contours by area * (1 - solidity)^k instead of plain area
//   a - learn skin colour from the face and back-project (calibrated mask)
// The face cascade only runs every few frames and the result is cached.
class ClassicalDetector {
public:
    ClassicalDetector();

    DetectionResult detect(const cv::Mat& frame);

    // runtime knobs
    int  v_min             = 60;
    bool use_face_mask     = true;
    bool use_shape_score   = true;
    bool use_adaptive_skin = true;

    // Back-projection cutoff (0..255). Lower picks up more hand and more
    // background. Tune live with , and .
    int  backproj_threshold = 35;

private:
    cv::CascadeClassifier face_cascade_;

    // cached so the cascade needn't run every frame
    std::vector<cv::Rect> cached_faces_;
    int                   detect_calls_ = 0;

    cv::Mat skin_hist_;       // H-S histogram, EMA-updated from the face
    int     trained_frames_ = 0;
};

}  // namespace gd
