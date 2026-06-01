#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

BIN="${ROOT_DIR}/build/bin/gesture_recognition"
if [[ ! -x "${BIN}" ]]; then
    echo "Binary not found: ${BIN}" >&2
    echo "Run scripts/build.sh first." >&2
    exit 1
fi

cd "${ROOT_DIR}"
"${BIN}" --model models/gesture_classifier.onnx "$@"
