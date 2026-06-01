"""Export the HaGRID YOLOv10n hand/gesture detector to ONNX.

HaGRID publishes YOLOv10n_hands.pt as an ultralytics-format checkpoint.
We load it with the ultralytics package and invoke its built-in ONNX
exporter, which produces a graph the C++ runtime consumes via
cv::dnn::readNetFromONNX.

Note on NMS: YOLOv10's architecture is NMS-free by design (it uses a
one-to-one assignment head during training), so the exported graph
already emits final detections — no postprocess NMS pass is needed on
the C++ side.

Note on classes: despite the file name, this model has 34 output classes
(33 HaGRID gestures + no_gesture). Used as a hand detector it just
returns the top-scoring box per frame, ignoring the predicted gesture
class. Used as a single-stage gesture detector, the class is the answer.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Export the HaGRID YOLOv10n detector to ONNX.")
    p.add_argument("--checkpoint", default=str(repo_root / "models" / "YOLOv10n_hands.pt"))
    p.add_argument("--out", default=str(repo_root / "models" / "hand_detector.onnx"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    print(f"loading checkpoint: {ckpt_path}")
    model = YOLO(str(ckpt_path))

    # ultralytics writes <stem>.onnx next to the checkpoint. We move it to
    # the requested --out path afterward so the caller controls naming.
    produced = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False,  # cv::dnn handles fixed shapes more reliably
        simplify=True,
    )
    produced_path = Path(produced)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if produced_path.resolve() != out_path.resolve():
        shutil.move(str(produced_path), out_path)

    print(f"wrote ONNX: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"classes: {len(model.names)} (model.names: {model.names})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
