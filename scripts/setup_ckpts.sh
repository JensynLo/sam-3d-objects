#!/usr/bin/env bash
# Populate ./ckpts from the local Hugging Face cache (hard links when possible, no
# extra disk space), downloading first if `hf`/`huggingface-cli` is available and
# the snapshot is missing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

snapshot() {  # $1 = repo id -> prints snapshot dir
  local dir="$HUB/models--${1//\//--}/snapshots"
  if ! ls "$dir"/*/ >/dev/null 2>&1; then
    echo "downloading $1 ..." >&2
    (command -v hf >/dev/null && hf download "$1" >&2) || huggingface-cli download "$1" >&2
  fi
  ls -d "$dir"/*/ | head -1
}

link() {  # $1 = src file, $2 = dst file
  mkdir -p "$(dirname "$2")"
  [ -s "$2" ] && return
  src="$(readlink -f "$1")"
  ln -f "$src" "$2" 2>/dev/null || cp "$src" "$2"
}

# Already populated (e.g. copied from another machine)? Then nothing to do.
if python3 -c "import sys; sys.path.insert(0, '$ROOT'); from sam3d_objects.paths import CheckpointPaths; CheckpointPaths().check()" 2>/dev/null; then
  echo "ckpts already complete"; du -shL "$ROOT"/ckpts/*; exit 0
fi

SAM3D="$(snapshot facebook/sam-3d-objects)checkpoints"
for f in pipeline.yaml ss_generator slat_generator ss_decoder slat_decoder_gs slat_decoder_mesh; do
  if [ "$f" = pipeline.yaml ]; then link "$SAM3D/$f" "$ROOT/ckpts/sam3d/$f"; continue; fi
  link "$SAM3D/$f.yaml" "$ROOT/ckpts/sam3d/$f.yaml"
  link "$SAM3D/$f.ckpt" "$ROOT/ckpts/sam3d/$f.ckpt"
done
link "$(snapshot Ruicheng/moge-vitl)model.pt" "$ROOT/ckpts/moge/model.pt"
du -shL "$ROOT"/ckpts/*
