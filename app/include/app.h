#pragma once

/// @file app.h
/// @brief Top-level application: owns pipeline modules, runs the live loop.

#include "config.h"

namespace gr {

/// Owns the camera, detector, classifier, renderer; runs the webcam loop.
class App {
public:
    explicit App(AppConfig cfg);
    ~App();

    App(const App&)            = delete;
    App& operator=(const App&) = delete;

    /// Real-time webcam loop. ESC to quit.
    void runLive();

    const AppConfig& config() const { return cfg_; }

private:
    AppConfig cfg_;
};

}  // namespace gr
