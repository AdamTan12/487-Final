"""Train the gesture classifier (MobileNetV3-small).

Reads a YAML config (see configs/mobilenetv3_small.yaml), fine-tunes the
ImageNet-pretrained backbone on the 9-class HaGRID subset, logs to
TensorBoard, saves checkpoints to `checkpoints/`.

Not implemented yet.
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the gesture classifier.")
    p.add_argument("--config", required=True,
                   help="Path to a training YAML config.")
    p.add_argument("--resume", default=None,
                   help="Optional checkpoint to resume from.")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    return p.parse_args()


def main() -> int:
    _ = parse_args()
    print("train.py: not implemented", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
