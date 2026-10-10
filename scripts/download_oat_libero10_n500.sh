#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-data/libero}"
ZIP_PATH="${ZIP_PATH:-${OUT_DIR}/libero10_N500.zarr.zip}"
ZARR_DIR="${ZARR_DIR:-${OUT_DIR}/libero10_N500.zarr}"
URL="${URL:-https://huggingface.co/datasets/chaoqi-liu/libero10_N500.zarr/resolve/main/libero10_N500.zarr.zip?download=true}"

mkdir -p "${OUT_DIR}"
if [ -d "${ZARR_DIR}" ]; then
  echo "exists: ${ZARR_DIR}"
  exit 0
fi

if [ ! -f "${ZIP_PATH}" ]; then
  echo "downloading ${URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -L "${URL}" -o "${ZIP_PATH}"
  else
    wget -O "${ZIP_PATH}" "${URL}"
  fi
fi

echo "extracting ${ZIP_PATH}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Neither python nor python3 was found for extraction." >&2
    exit 1
  fi
fi
export ZIP_PATH OUT_DIR
"${PYTHON_BIN}" - <<'PY'
import os
import zipfile
from pathlib import Path

zip_path = Path(os.environ["ZIP_PATH"])
out_dir = Path(os.environ["OUT_DIR"])
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
print(out_dir / "libero10_N500.zarr")
PY
