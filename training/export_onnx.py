"""Export the trained PyTorch model to ONNX.

Loads a checkpoint, wraps it in an eval-mode module, traces with a dummy
1x3x224x224 input, writes ONNX (opset 17) to the given path. The exported
model is the artifact the C++ runtime consumes via cv::dnn::readNetFromONNX.

Not implemented yet.
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the classifier to ONNX.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out",        default="../models/gesture_classifier.onnx")
    p.add_argument("--opset",      type=int, default=17)
    return p.parse_args()


def main() -> int:
    _ = parse_args()
    print("export_onnx.py: not implemented", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
