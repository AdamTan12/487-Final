#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

mkdir -p "${ROOT_DIR}/build"
cd "${ROOT_DIR}/build"

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

echo
echo "Build complete: ${ROOT_DIR}/build/bin/gesture_recognition"
