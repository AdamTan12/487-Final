#include "app.h"
#include "config.h"

#include <exception>
#include <iostream>
#include <utility>

int main(int argc, char** argv) {
    try {
        gr::AppConfig cfg = gr::parseArgs(argc, argv);
        gr::App app(std::move(cfg));
        app.runLive();
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
