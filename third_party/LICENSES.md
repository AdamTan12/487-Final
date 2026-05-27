# Third-Party Licenses

Dependencies used by this project, with attribution to be reproduced in
the writeup and any binary distribution.

## Runtime (C++ application)

### OpenCV 4
- License: Apache License 2.0
- Project: https://opencv.org/
- Usage: image I/O, video capture, classical CV (HSV, contours,
  morphology), ONNX inference via `cv::dnn`.

## Offline (training pipeline)

### PyTorch
- License: BSD 3-Clause "New" or "Revised"
- Project: https://pytorch.org/
- Usage: model definition, fine-tuning loop, ONNX export.

### TorchVision
- License: BSD 3-Clause
- Project: https://pytorch.org/vision/
- Usage: MobileNetV3-small backbone with ImageNet weights, image transforms.

### ONNX / ONNX Runtime
- License: MIT (ONNX), MIT (ONNX Runtime)
- Project: https://onnx.ai/ , https://onnxruntime.ai/
- Usage: export format and verification of exported models.

### Other Python deps
- NumPy (BSD 3-Clause), Pillow (HPND), opencv-python (Apache 2.0),
  scikit-learn (BSD 3-Clause), matplotlib (PSF-based), tensorboard
  (Apache 2.0), PyYAML (MIT), tqdm (MIT/MPL 2.0).

## Datasets and pretrained models

### HaGRID (HAnd Gesture Recognition Image Dataset)
- License: Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0)
- Project: https://github.com/hukenovs/hagrid
- Usage: training/test imagery for the 9-class gesture subset (`like`,
  `dislike`, `peace`, `mute`, `stop`, `gun` (HaGRID `three_gun`), `ok`,
  `fist`, `palm`). We may also use HaGRID's pretrained YOLOv10n hand
  detector as the ONNX-detector fallback at runtime.
- Note: under CC BY-SA 4.0, derived models/data we distribute are
  subject to the share-alike clause.

### MobileNetV3 (architecture)
- Reference: Howard et al., "Searching for MobileNetV3" (2019),
  https://arxiv.org/abs/1905.02244
- Implementation source: `torchvision.models.mobilenet_v3_small` with
  ImageNet-pretrained weights (BSD 3-Clause via PyTorch).
