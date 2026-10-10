#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REVISION=6090ff21837566fed47b7c9061c9899f4749d36b
if [[ ! -d thirdparty/LIBERO ]]; then
  git clone https://github.com/Chaoqi-LIU/LIBERO.git thirdparty/LIBERO
  git -C thirdparty/LIBERO checkout --detach "$REVISION"
fi
if [[ "$(git -C thirdparty/LIBERO rev-parse HEAD)" != "$REVISION" ]]; then
  echo "Unexpected LIBERO revision; use a fresh checkout." >&2
  exit 1
fi
python -m pip install cmake==4.1.3
python -m pip install -r requirements-sim.txt
python -m pip install --no-deps -e thirdparty/LIBERO
python -m pip install --no-deps -e .
python -m pip check
echo "DSP diffusion and LIBERO installed. No OAT checkout is required."
