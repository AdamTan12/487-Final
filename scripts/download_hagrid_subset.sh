#!/usr/bin/env bash
# Download the 9-gesture HaGRID subset used by this project.
#
# The bulk of the gestures live in HaGRID's v1 bucket; "three_gun" (which
# we expose as the "gun" class) lives in the HaGRID v2 bucket. We download
# everything into data/hagrid_raw/ and skip files that are already present
# so the script is safe to re-run.
#
# Total download is on the order of tens of GB. Plan disk + time accordingly.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
RAW_DIR="${ROOT_DIR}/data/hagrid_raw"

V1_BASE="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset"
V2_BASE="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagrid_v2_zip"

# 8 gestures from HaGRID v1.
V1_GESTURES=(like dislike peace mute stop ok fist palm)
# 1 gesture from HaGRID v2 (HaGRID's "three_gun" maps to our "gun" class).
V2_GESTURES=(three_gun)

mkdir -p "${RAW_DIR}"

fetch() {
    local url="$1"
    local dest="$2"
    if [[ -f "${dest}" ]]; then
        echo "[skip] ${dest##*/} already downloaded"
        return 0
    fi
    echo "[get ] ${url}"
    curl -fL --retry 3 --retry-delay 5 -o "${dest}.partial" "${url}"
    mv "${dest}.partial" "${dest}"
}

for g in "${V1_GESTURES[@]}"; do
    fetch "${V1_BASE}/${g}.zip" "${RAW_DIR}/${g}.zip"
done

for g in "${V2_GESTURES[@]}"; do
    fetch "${V2_BASE}/${g}.zip" "${RAW_DIR}/${g}.zip"
done

# Annotations (small) — needed by prepare_data.py to crop hand regions.
fetch "${V1_BASE}/ann_train_val.zip" "${RAW_DIR}/ann_train_val.zip"
fetch "${V1_BASE}/ann_test.zip"      "${RAW_DIR}/ann_test.zip"

echo
echo "Downloaded HaGRID subset into ${RAW_DIR}"
du -sh "${RAW_DIR}" || true
echo "Next step: python training/prepare_data.py"
