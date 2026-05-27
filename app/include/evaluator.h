#pragma once

/// @file evaluator.h
/// @brief Offline evaluation: walk a labeled test directory, score the pipeline.

#include <string>

namespace gr {

class App;  // forward decl — Evaluator holds a non-owning ref to the pipeline

/// Runs the full pipeline against a labeled directory of test images.
///
/// Expected layout:
///   test_dir/
///     like/    *.jpg
///     dislike/ *.jpg
///     ...      (one subdir per class name in models/class_names.txt)
///
/// Writes a row per image to `output_csv` with header
/// `image_path,expected_class,predicted_class,confidence,correct`,
/// and prints overall accuracy + a 9x9 confusion matrix to stdout.
class Evaluator {
public:
    explicit Evaluator(App& app);

    void evaluate(const std::string& test_dir, const std::string& output_csv);

private:
    App& app_;
};

}  // namespace gr
