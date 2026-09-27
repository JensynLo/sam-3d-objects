#!/usr/bin/env python
"""SAM 3D Objects: single image -> 3D Gaussians (.ply) or mesh (.glb).

    python infer.py <image> <3dgs|mesh> <output_dir> [--mask mask.png] [--seed 42]

The object mask is taken from --mask if given, otherwise from the alpha channel
of the image. All weights are read from ./ckpts (see scripts/setup_ckpts.sh).
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights come from ./ckpts only

import numpy as np
from PIL import Image

FORMAT_ALIASES = {"3dgs": "gaussian", "gs": "gaussian", "gaussian": "gaussian", "mesh": "mesh"}


def load_mask(mask_path: Path, size) -> np.ndarray:
    """Binary mask from the alpha channel if the file has one (official demo masks),
    otherwise from its luminance, thresholded at half of its maximum."""
    mask = Image.open(mask_path)
    mask = mask.getchannel("A") if "A" in mask.getbands() else mask.convert("L")
    mask = np.array(mask.resize(size, Image.NEAREST))
    return mask > mask.max() / 2


def load_rgba(image_path: Path, mask_path=None) -> np.ndarray:
    image = Image.open(image_path)
    if mask_path is not None:
        mask = load_mask(mask_path, image.size)
    elif image.mode in ("RGBA", "LA") or "transparency" in image.info:
        mask = np.array(image.convert("RGBA"))[..., 3] > 0
    else:
        sys.exit(f"{image_path} has no alpha channel: pass the object mask with --mask")
    if not mask.any():
        sys.exit("object mask is empty")
    rgb = np.array(image.convert("RGB"))
    return np.concatenate([rgb, mask[..., None].astype(np.uint8) * 255], axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="input image (RGBA, or RGB together with --mask)")
    parser.add_argument("format", choices=sorted(FORMAT_ALIASES), help="output representation")
    parser.add_argument("output_dir", type=Path, help="directory the result is written to")
    parser.add_argument("--mask", type=Path, default=None, help="binary object mask (default: image alpha)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_format = FORMAT_ALIASES[args.format]
    rgba = load_rgba(args.image, args.mask)

    from sam3d_objects.pipeline import export
    from sam3d_objects.pipeline.inference import Sam3DPipeline

    result = Sam3DPipeline().run(rgba, output_format, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "gaussian":
        out = export.save_gaussian(result, args.output_dir / f"{args.image.stem}.ply")
    else:
        out = export.save_mesh(result, args.output_dir / f"{args.image.stem}.glb")
    print(f"Saved {output_format} to {out}")


if __name__ == "__main__":
    main()
