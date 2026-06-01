#include "config.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace gr {

namespace {

void printUsage(std::ostream& os, const char* prog) {
    os << "usage: " << prog << " [options]\n"
       << "  --model <path>           classifier ONNX path\n"
       << "  --detector-model <path>  detector ONNX path\n"
       << "  --camera <int>           camera index (default: 0)\n"
       << "  --debug                  enable debug overlays/logging\n"
       << "  -h, --help               show this help\n";
}

[[noreturn]] void die(const char* prog, const std::string& msg) {
    std::cerr << "error: " << msg << "\n\n";
    printUsage(std::cerr, prog);
    std::exit(2);
}

std::string requireValue(const char* prog,
                         int argc, char** argv,
                         int& i, const char* flag) {
    if (i + 1 >= argc) {
        die(prog, std::string("missing value for ") + flag);
    }
    return std::string(argv[++i]);
}

}  // namespace

AppConfig parseArgs(int argc, char** argv) {
    const char* prog = (argc > 0 && argv[0]) ? argv[0] : "gesture_recognition";
    AppConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(std::cout, prog);
            std::exit(0);
        } else if (arg == "--model") {
            cfg.model_path = requireValue(prog, argc, argv, i, "--model");
        } else if (arg == "--detector-model") {
            cfg.detector_model_path = requireValue(prog, argc, argv, i, "--detector-model");
        } else if (arg == "--camera") {
            cfg.camera_index = std::stoi(requireValue(prog, argc, argv, i, "--camera"));
        } else if (arg == "--debug") {
            cfg.debug = true;
        } else {
            die(prog, "unknown argument: " + arg);
        }
    }

    return cfg;
}

}  // namespace gr
