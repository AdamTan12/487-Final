# Gesture Recognition

Real-time hand gesture recognition using OpenCV (C++) for the runtime
application and PyTorch (Python) for offline model training.

The **C++ application is the graded artifact.** Python is offline-only
tooling that produces an ONNX model the C++ app consumes at runtime.

## Team

- TBD
- TBD

## Gestures

Nine one-handed gestures (HaGRID subset). Class indices are fixed and
must match between training and inference (see `models/class_names.txt`):

| idx | gesture | HaGRID class |
| --: | ------- | ------------ |
| 0   | like    | `like`       |
| 1   | dislike | `dislike`    |
| 2   | peace   | `peace`      |
| 3   | mute    | `mute`       |
| 4   | stop    | `stop`       |
| 5   | gun     | `three_gun`  |
| 6   | ok      | `ok`         |
| 7   | fist    | `fist`       |
| 8   | palm    | `palm`       |

## Directory layout

```
app/          C++ application (graded artifact). OpenCV-only.
  include/    Public headers, all in namespace gr::
  src/        Implementations
training/     Python offline pipeline: data prep, training, ONNX export
models/       Trained ONNX model + preprocessing contract + class names
data/         HaGRID raw + processed (gitignored; not committed)
tests/        Evaluation fixtures and expected results
scripts/      Build and run helpers for Windows / macOS / Linux
docs/         Design notes and writeup
third_party/  Vendor licenses we need to cite
```

## Runtime modes

The C++ binary supports three modes via `--mode`:

- `live` (default) — open webcam, run detector + classifier, render overlay.
- `eval` — walk `tests/test_images/{class}/*.jpg`, run the full pipeline,
  write per-image predictions to a CSV, print accuracy + confusion matrix.
- `bench` — measure end-to-end FPS and per-stage timings on the webcam.

Other flags:

- `--detector classical|onnx` — choose hand detector. Classical (HSV +
  contours) is primary; ONNX (YOLOv10n from HaGRID) is the fallback.
- `--model <path>` — path to the classifier ONNX (default
  `models/gesture_classifier.onnx`).
- `--camera <int>` — camera index (default 0).
- `--debug` — extra overlays and console logging.

## Build

### macOS

Requires OpenCV 4 (`brew install opencv`) and CMake 3.16+.

```bash
bash scripts/build.sh
./build/bin/gesture_recognition
```

### Windows

Requires Visual Studio 2022 with the "Desktop development with C++" workload
and OpenCV 4 (either pre-built or via vcpkg). Set `OpenCV_DIR` to the
directory containing `OpenCVConfig.cmake` if CMake can't find OpenCV.

```bat
scripts\build.bat
build\bin\Release\gesture_recognition.exe
```

## Training pipeline (offline, Python)

Python is **not** invoked at runtime. The end-to-end offline flow is:

```bash
cd training
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

bash ../scripts/download_hagrid_subset.sh            # ~tens of GB
python prepare_data.py                               # crops + train/val/test split
python train.py    --config configs/mobilenetv3_small.yaml
python evaluate.py --config configs/mobilenetv3_small.yaml
python export_onnx.py --out ../models/gesture_classifier.onnx
python verify_onnx.py --model ../models/gesture_classifier.onnx
```

The exported ONNX, along with `models/class_names.txt` and
`models/preprocessing.txt`, is the complete handoff to the C++ runtime.
Any change to preprocessing in Python MUST be mirrored in C++ or accuracy
will collapse silently.
