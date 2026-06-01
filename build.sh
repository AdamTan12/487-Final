#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
mkdir -p "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build"

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

echo
echo "Built: ${SCRIPT_DIR}/build/bin/gesture_demo"
