"""Single source of truth for where weights live: `<project>/ckpts`.

    ckpts/
      sam3d/   facebook/sam-3d-objects `checkpoints/` folder (pipeline.yaml, *.yaml, *.ckpt)
      moge/    Ruicheng/moge-vitl `model.pt`

Nothing else is loaded at runtime: the DINOv2 backbones are part of
ss_generator.ckpt / slat_generator.ckpt, and MoGe's own DINOv2 is part of model.pt.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = PROJECT_ROOT / "ckpts"

_SAM3D_MODELS = ("ss_generator", "slat_generator", "ss_decoder", "slat_decoder_gs", "slat_decoder_mesh")


class CheckpointPaths:
    def __init__(self, root=CKPT_ROOT):
        self.root = Path(root)
        self.moge = self.root / "moge" / "model.pt"

    def sam3d(self, filename: str) -> str:
        return str(self.root / "sam3d" / filename)

    def required_files(self) -> list:
        files = [self.sam3d("pipeline.yaml"), str(self.moge)]
        for name in _SAM3D_MODELS:
            files += [self.sam3d(f"{name}.yaml"), self.sam3d(f"{name}.ckpt")]
        return [Path(f) for f in files]

    def check(self):
        missing = [str(f) for f in self.required_files() if not f.is_file() or f.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(
                "Missing checkpoint files (run scripts/setup_ckpts.sh):\n  " + "\n  ".join(missing)
            )
