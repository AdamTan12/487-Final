#include "app.h"
#include "config.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        gr::AppConfig cfg = gr::parseArgs(argc, argv);
        gr::App app(std::move(cfg));

        switch (app.config().mode) {
            case gr::AppMode::Live:
                app.runLive();
                break;
            case gr::AppMode::Eval:
                app.runEval(app.config().eval_dir, app.config().eval_output_csv);
                break;
            case gr::AppMode::Bench:
                app.runBench();
                break;
        }
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
