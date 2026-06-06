#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root so assets/emoji/*.png resolve relative to the cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

EXE="build/bin/gesture_demo"
if [[ ! -x "${EXE}" ]]; then
    echo "${EXE} not found. Build it first with ./build.sh"
    exit 1
fi

# On macOS/Linux the OpenCV libs are found automatically (Homebrew/apt set the
# loader path), so no PATH juggling is needed here.
exec "./${EXE}"
