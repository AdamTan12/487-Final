"""Prepare HaGRID subset for training.

Walks `data/hagrid_raw/`, uses HaGRID's annotation JSONs to crop each
hand region with a small padding margin, then writes cropped images into
`data/hagrid_processed/{train,val,test}/{class}/`. Class names follow
`models/class_names.txt`; the HaGRID class `three_gun` is renamed to
`gun` on disk.

Not implemented yet — fill in over the next sprint.
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crop and split the HaGRID subset.")
    p.add_argument("--raw_dir",       default="../data/hagrid_raw",
                   help="Source HaGRID archive root.")
    p.add_argument("--processed_dir", default="../data/hagrid_processed",
                   help="Destination for cropped + split images.")
    p.add_argument("--val_frac",      type=float, default=0.1)
    p.add_argument("--test_frac",     type=float, default=0.1)
    p.add_argument("--padding",       type=float, default=0.15,
                   help="Bbox expansion ratio before cropping.")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main() -> int:
    _ = parse_args()
    print("prepare_data.py: not implemented", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
