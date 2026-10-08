# SAM 3D Objects — minimal inference

Inference-only refactor of [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects):
one image (+ object mask) in, 3D Gaussians or a mesh out. Training, evaluation, layout
post-optimisation, rendering, texture baking and the notebook/demo code are removed.

## Usage

```bash
./scripts/setup_env.sh                         # once: conda env `sam3d` with pinned deps
./scripts/setup_ckpts.sh                       # once: fill ./ckpts from the HF cache (downloads if missing;
                                               #       facebook/sam-3d-objects is gated -> `hf auth login` first)
conda activate sam3d
python infer.py examples/teddy.png mesh outputs/    # -> outputs/teddy.glb
python infer.py examples/teddy.png 3dgs outputs/    # -> outputs/teddy.ply
python infer.py <image> <3dgs|mesh> <output_dir> [--mask mask.png] [--seed 42]
```

| argument     | meaning                                                                   |
|--------------|---------------------------------------------------------------------------|
| `image`      | input image. The object mask is its alpha channel, unless `--mask` is given |
| `3dgs\|mesh` | `3dgs` -> `<output_dir>/<image stem>.ply` (standard 3DGS ply), `mesh` -> `<output_dir>/<image stem>.glb` (vertex colours) |
| `output_dir` | created if missing                                                        |

The result is placed in the camera frame by the stage-1 pose (rotation / translation / scale,
decoded as upstream's `ScaleShiftInvariant` pose decoder and applied like the demo's `make_scene`):
PyTorch3D camera convention, x left, y up, z forward, metric units of the MoGe pointmap.
`Sam3DPipeline.run(..., apply_pose=False)` keeps the canonical frame (unit cube at the origin).
The mesh is the raw FlexiCubes extraction with vertex colours, i.e. exactly what the
official demo returns (`with_mesh_postprocess=False, with_texture_baking=False`).

## Checkpoints

Every weight is read from `./ckpts` (`sam3d_objects/paths.py` is the only place that knows
about it); nothing touches the HF hub, torch hub or GitHub at runtime (`infer.py` sets
`HF_HUB_OFFLINE=1`).

```
ckpts/
  sam3d/   pipeline.yaml + {ss_generator,slat_generator,ss_decoder,slat_decoder_gs,slat_decoder_mesh}.{yaml,ckpt}
           (from hf: facebook/sam-3d-objects, folder checkpoints/)
  moge/    model.pt   (from hf: Ruicheng/moge-vitl)
```

| load                         | file                                   | note |
|------------------------------|----------------------------------------|------|
| MoGe depth / pointmap        | `ckpts/moge/model.pt`                  | its DINOv2 backbone is built with `pretrained=False` and filled from model.pt |
| stage-1 flow + cond embedder | `ckpts/sam3d/ss_generator.ckpt`        | contains both DINOv2-L/14-reg backbones, so no separate DINOv2 download; read once, split by prefix |
| stage-1 occupancy decoder    | `ckpts/sam3d/ss_decoder.ckpt`          | |
| stage-2 flow + cond embedder | `ckpts/sam3d/slat_generator.ckpt`      | same as stage 1 |
| output decoder               | `ckpts/sam3d/slat_decoder_{gs,mesh}.ckpt` | only the requested one is loaded |

All `load_state_dict` calls are `strict=True`. DINOv2 is vendored
(`sam3d_objects/model/backbone/dit/embedder/dinov2`) instead of `torch.hub.load`.

## Behaviour vs. upstream

Same preprocessing, samplers, hyper-parameters (read from `ckpts/sam3d/pipeline.yaml`) and
RNG stream as `notebook/inference.py::Inference.__call__`. Models are loaded stage by stage
and freed afterwards, so peak GPU memory is ~one stage instead of everything at once
(upstream does not fit a 12 GB card). The `sam3d_objects` package/module paths are kept so
the released yaml `_target_`s work unmodified.
