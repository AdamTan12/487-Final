#include "preprocess.h"

#include <stdexcept>

namespace gr {

cv::Mat prepareBlob(const cv::Mat&    /*frame*/,
                    const cv::Rect&   /*bbox*/,
                    const AppConfig&  /*cfg*/) {
    throw std::runtime_error("not implemented: prepareBlob");
}

}  // namespace gr
