"""Export the HaGRID ResNet18 gesture classifier to ONNX.

Loads HaGRID's published ResNet18.pth (a vanilla torchvision ResNet18
with a 34-class `fc` head — 33 gestures + no_gesture), traces with a
dummy 1x3x224x224 input, writes ONNX (opset 17). The exported model is
the artifact the C++ runtime consumes via cv::dnn::readNetFromONNX.

Checkpoint format (from HaGRID's training loop):
    {
        "MODEL_STATE":     <state_dict>,
        "OPTIMIZER_STATE": ...,
        ...
    }
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.models as tvm

NUM_GESTURES = 34  # HaGRIDv2: 33 gestures + no_gesture.


def _extract_state_dict(checkpoint) -> dict:
    """Unwrap HaGRID's `MODEL_STATE` key (and common alternates)."""
    if isinstance(checkpoint, dict):
        for key in ("MODEL_STATE", "state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Export the HaGRID ResNet18 classifier to ONNX.")
    p.add_argument("--checkpoint", default=str(repo_root / "models" / "ResNet18.pth"))
    p.add_argument("--out", default=str(repo_root / "models" / "gesture_classifier.onnx"))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--num-gestures", type=int, default=NUM_GESTURES)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    print(f"loading checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(raw)

    model = tvm.resnet18(weights=None, num_classes=args.num_gestures)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"warning: missing={missing} unexpected={unexpected}", file=sys.stderr)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        sanity = model(dummy)
    print(f"forward sanity check ok, output shape: {tuple(sanity.shape)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["gesture_logits"],
        dynamic_axes={"input": {0: "batch"}, "gesture_logits": {0: "batch"}},
    )
    print(f"wrote ONNX: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
