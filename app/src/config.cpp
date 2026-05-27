#include "config.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace gr {

namespace {

void printUsage(std::ostream& os, const char* prog) {
    os << "usage: " << prog << " [options]\n"
       << "  --mode {live|eval|bench}     (default: live)\n"
       << "  --detector {classical|onnx}  (default: classical)\n"
       << "  --model <path>               classifier ONNX path\n"
       << "  --detector-model <path>      detector ONNX path (if --detector onnx)\n"
       << "  --camera <int>               camera index (default: 0)\n"
       << "  --eval-dir <path>            test image root (eval mode)\n"
       << "  --eval-out <path>            output CSV (eval mode)\n"
       << "  --debug                      enable debug overlays/logging\n"
       << "  -h, --help                   show this help\n";
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
        } else if (arg == "--mode") {
            const auto v = requireValue(prog, argc, argv, i, "--mode");
            if      (v == "live")  cfg.mode = AppMode::Live;
            else if (v == "eval")  cfg.mode = AppMode::Eval;
            else if (v == "bench") cfg.mode = AppMode::Bench;
            else die(prog, "unknown --mode: " + v);
        } else if (arg == "--detector") {
            const auto v = requireValue(prog, argc, argv, i, "--detector");
            if      (v == "classical") cfg.detector = DetectorType::Classical;
            else if (v == "onnx")      cfg.detector = DetectorType::Onnx;
            else die(prog, "unknown --detector: " + v);
        } else if (arg == "--model") {
            cfg.model_path = requireValue(prog, argc, argv, i, "--model");
        } else if (arg == "--detector-model") {
            cfg.detector_model_path = requireValue(prog, argc, argv, i, "--detector-model");
        } else if (arg == "--camera") {
            cfg.camera_index = std::stoi(requireValue(prog, argc, argv, i, "--camera"));
        } else if (arg == "--eval-dir") {
            cfg.eval_dir = requireValue(prog, argc, argv, i, "--eval-dir");
        } else if (arg == "--eval-out") {
            cfg.eval_output_csv = requireValue(prog, argc, argv, i, "--eval-out");
        } else if (arg == "--debug") {
            cfg.debug = true;
        } else {
            die(prog, "unknown argument: " + arg);
        }
    }

    return cfg;
}

}  // namespace gr
