"""Sanity-check an exported ONNX model.

Loads the ONNX via onnxruntime, runs a few sample images through it, and
compares predictions against the original PyTorch checkpoint. Fails loudly
if the per-class probabilities disagree beyond a small tolerance — that
means the C++ runtime would also disagree.

Not implemented yet.
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify exported ONNX matches PyTorch.")
    p.add_argument("--model",      required=True,
                   help="Path to the exported ONNX file.")
    p.add_argument("--checkpoint", default=None,
                   help="Optional torch checkpoint to compare against.")
    p.add_argument("--samples",    default="../data/hagrid_processed/test",
                   help="Directory of images to test on.")
    p.add_argument("--tolerance",  type=float, default=1.0e-4)
    return p.parse_args()


def main() -> int:
    _ = parse_args()
    print("verify_onnx.py: not implemented", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
