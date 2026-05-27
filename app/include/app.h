#pragma once

/// @file app.h
/// @brief Top-level application: owns all pipeline modules, dispatches per mode.

#include "config.h"

#include <string>

namespace gr {

/// Owns the camera, detector, classifier, smoother, renderer, etc. and
/// runs one of three modes (live / eval / bench).
class App {
public:
    explicit App(AppConfig cfg);
    ~App();

    App(const App&)            = delete;
    App& operator=(const App&) = delete;

    /// Real-time webcam loop. ESC to quit.
    void runLive();

    /// Walk `test_dir`, run pipeline, write `out_csv`, print metrics.
    void runEval(const std::string& test_dir, const std::string& out_csv);

    /// Time end-to-end FPS and per-stage latency.
    void runBench();

    const AppConfig& config() const { return cfg_; }

private:
    AppConfig cfg_;
};

}  // namespace gr
