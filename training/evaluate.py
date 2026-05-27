"""Evaluate a trained classifier checkpoint.

Loads a checkpoint, runs it over the held-out test split, prints overall
accuracy + per-class precision/recall/F1 + a 9x9 confusion matrix.
Optionally writes a PNG of the confusion matrix.

Not implemented yet.
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained classifier.")
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir",    default="runs/eval")
    return p.parse_args()


def main() -> int:
    _ = parse_args()
    print("evaluate.py: not implemented", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
