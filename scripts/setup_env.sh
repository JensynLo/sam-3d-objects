#!/usr/bin/env bash
# Create the `sam3d` conda env and install every runtime dependency.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-sam3d}"
eval "$(conda shell.bash hook)"
conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"
pip install -r "$ROOT/requirements.txt"
python - <<'PY'
import torch, spconv.pytorch, xformers, moge, utils3d
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
