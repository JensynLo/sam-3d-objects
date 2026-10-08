"""Single-image -> 3D Gaussians / mesh inference for SAM 3D Objects.

Behaviourally equivalent to the released `InferencePipelinePointMap.run(...)` as
called by the official demo (`notebook/inference.py::Inference.__call__`), with
everything that is not needed for that one code path removed.

Differences that are deliberate:
  * every weight is read from `<project>/ckpts` (see `sam3d_objects.paths`);
    nothing is fetched from the HF hub / torch hub.
  * models are loaded stage by stage and released afterwards, so the peak GPU
    memory is one stage (~7 GB) instead of all of them at once.
  * only the decoder of the requested output format is loaded.
"""

import gc
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from sam3d_objects import paths
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import get_mask
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from sam3d_objects.pipeline import pose as pose_utils
from sam3d_objects.pipeline.sparse_structure import (
    downsample_sparse_structure,
    prune_sparse_structure,
)

FORMATS = ("gaussian", "mesh")

# Defaults of the released InferencePipeline.__init__ that pipeline.yaml does not override.
_DEFAULTS = dict(
    ss_inference_steps=25,
    ss_rescale_t=3,
    ss_cfg_strength=7,
    ss_cfg_interval=[0, 500],
    slat_inference_steps=25,
    slat_rescale_t=3,
    slat_cfg_strength=5,
    slat_cfg_interval=[0, 500],
    downsample_ss_dist=0,
    dtype="bfloat16",
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# MoGe (OpenCV-style camera: x right, y down, z forward) -> PyTorch3D camera
# (x left, y up, z forward). Same matrix the released code builds with
# pytorch3d.look_at_view_transform(eye=(0,0,-1), up=(0,-1,0)).
_MOGE_TO_P3D = torch.tensor([-1.0, -1.0, 1.0])


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    out = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not out:
        raise KeyError(f"no weights with prefix {prefix!r} in checkpoint")
    return out


class Sam3DPipeline:
    def __init__(self, ckpt_root=paths.CKPT_ROOT, device: str = "cuda"):
        self.ckpts = paths.CheckpointPaths(ckpt_root)
        self.ckpts.check()
        self.device = torch.device(device)

        cfg = OmegaConf.load(self.ckpts.sam3d("pipeline.yaml"))
        self.cfg = cfg
        opt = {k: cfg.get(k, v) for k, v in _DEFAULTS.items()}
        self.opt = opt
        self.dtype = _DTYPES[opt["dtype"]]
        self.ss_preprocessor = instantiate(cfg.ss_preprocessor)
        self.slat_preprocessor = instantiate(cfg.slat_preprocessor)
        self.slat_mean = torch.tensor(list(cfg.slat_mean))
        self.slat_std = torch.tensor(list(cfg.slat_std))
        self._rng = None
        self.last_pose = None

    # ------------------------------------------------------------------ loading
    @contextmanager
    def _loaded(self, **builders):
        """Build the given models on GPU for the duration of the block, then free them."""
        models = {name: build() for name, build in builders.items()}
        try:
            yield models
        finally:
            models.clear()
            gc.collect()
            torch.cuda.empty_cache()

    @contextmanager
    def _sampling_rng(self):
        """The released pipeline seeds once, with every model already resident.
        Here models are built lazily (which consumes RNG), so the sampling RNG
        stream is carried across stages explicitly to keep results identical."""
        if self._rng is not None:
            torch.set_rng_state(self._rng[0])
            torch.cuda.set_rng_state(self._rng[1], self.device)
        yield
        self._rng = (torch.get_rng_state(), torch.cuda.get_rng_state(self.device))

    def _finalize(self, model: torch.nn.Module) -> torch.nn.Module:
        model.requires_grad_(False)
        return model.eval().to(self.device)

    def _load_depth_model(self):
        depth_cfg = self.cfg.depth_model.copy()
        depth_cfg.model.pretrained_model_name_or_path = str(self.ckpts.moge)
        logger.info(f"Loading depth model from {self.ckpts.moge}")
        return instantiate(depth_cfg, device=str(self.device))

    def _load_generator_stage(self, name: str):
        """`{name}.ckpt` holds both the flow generator and its condition embedder
        (incl. the DINOv2 backbones). Read the file once and split it by prefix."""
        cfg = OmegaConf.load(self.ckpts.sam3d(f"{name}.yaml"))["module"]
        ckpt_path = self.ckpts.sam3d(f"{name}.ckpt")
        logger.info(f"Loading {name} from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)["state_dict"]

        generator = instantiate(cfg["generator"]["backbone"])
        generator.load_state_dict(_strip_prefix(state, "_base_models.generator."), strict=True)
        embedder = instantiate(cfg["condition_embedder"]["backbone"])
        embedder.load_state_dict(_strip_prefix(state, "_base_models.condition_embedder."), strict=True)
        del state
        return self._finalize(generator), self._finalize(embedder)

    def _load_flat(self, name: str):
        """Models whose checkpoint is a bare state_dict and whose yaml is a flat `_target_` config."""
        cfg = OmegaConf.load(self.ckpts.sam3d(f"{name}.yaml"))
        cfg.pop("pretrained_ckpt_path", None)
        ckpt_path = self.ckpts.sam3d(f"{name}.ckpt")
        logger.info(f"Loading {name} from {ckpt_path}")
        model = instantiate(cfg)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True), strict=True)
        return self._finalize(model)

    # ------------------------------------------------------------ preprocessing
    def _preprocess(self, rgba: np.ndarray, preprocessor, pointmap=None) -> dict:
        rgba_t = torch.from_numpy(rgba.astype(np.float32) / 255).permute(2, 0, 1).contiguous()
        item = preprocessor._process_image_mask_pointmap_mess(
            rgba_t[:3], get_mask(rgba_t, None, "ALPHA_CHANNEL"), pointmap
        )
        keys = ["mask", "image", "rgb_image", "rgb_image_mask"]
        if pointmap is not None:
            keys += ["pointmap", "rgb_pointmap"]
        out = {k: item[k][None].to(self.device) for k in keys}
        # Normalisation moments of the pointmap: not a model input, needed to decode the pose.
        moments = {k: item[k][None].to(self.device) for k in ("pointmap_scale", "pointmap_shift") if k in item}
        return out, moments

    def compute_pointmap(self, rgba: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(rgba[..., :3].astype(np.float32) / 255).permute(2, 0, 1).contiguous()
        with self._loaded(depth=self._load_depth_model) as m:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
                points = m["depth"](image)["pointmaps"]  # (H, W, 3)
        points = points * _MOGE_TO_P3D.to(points)
        # MoGe returns +-inf where it sees no depth (e.g. a plain white background).
        # Everything downstream treats NaN as "no point" (nanmedian in the SSI moments,
        # NaN padding, the pointmap embedder's validity mask), but not inf: one inf-heavy
        # background makes the scene scale inf, zeroes the object points after
        # normalisation and makes the decoded pose inf.
        points = torch.where(points.isfinite(), points, torch.full_like(points, float("nan")))
        return points.permute(2, 0, 1)

    # ------------------------------------------------------------------- stages
    @torch.no_grad()
    def sample_sparse_structure(self, ss_input: dict):
        """Returns (coords, pose latents, downsample_factor)."""
        opt = self.opt
        builders = dict(
            stage=lambda: self._load_generator_stage("ss_generator"),
            decoder=lambda: self._load_flat("ss_decoder"),
        )
        with self._loaded(**builders) as m:
            (generator, embedder), decoder = m["stage"], m["decoder"]
            generator.no_shortcut = True
            generator.inference_steps = opt["ss_inference_steps"]
            generator.rescale_t = opt["ss_rescale_t"]
            generator.reverse_fn.strength = opt["ss_cfg_strength"]
            generator.reverse_fn.interval = list(opt["ss_cfg_interval"])
            generator.reverse_fn.unconditional_handling = "add_flag"

            bs = ss_input["image"].shape[0]
            latent_shapes = {
                k: (bs, v.pos_emb.shape[0], v.input_layer.in_features)
                for k, v in generator.reverse_fn.backbone.latent_mapping.items()
            }
            logger.info(f"Sampling sparse structure ({generator.inference_steps} steps)")
            with self._sampling_rng(), torch.autocast(device_type="cuda", dtype=self.dtype):
                cond = embedder(**ss_input)
                latents = generator(latent_shapes, self.device, cond)
                shape_latent = latents["shape"]
                pose_latents = {k: v.float() for k, v in latents.items() if k != "shape"}
                occupancy = decoder(
                    shape_latent.permute(0, 2, 1).contiguous().view(bs, 8, 16, 16, 16)
                )
            coords = torch.argwhere(occupancy > 0)[:, [0, 2, 3, 4]].int()
            del m, generator, embedder, decoder

        n_voxels = coords.shape[0]
        if n_voxels == 0:
            raise RuntimeError("Sparse structure is empty; check the input mask.")
        if opt["downsample_ss_dist"] > 0:
            coords = prune_sparse_structure(coords, max_neighbor_axes_dist=opt["downsample_ss_dist"])
        coords, downsample_factor = downsample_sparse_structure(coords)
        logger.info(
            f"Sparse structure: {n_voxels} voxels -> {coords.shape[0]} active coords"
            f" (downsample factor {downsample_factor})"
        )
        return coords, pose_latents, downsample_factor

    @torch.no_grad()
    def sample_slat(self, slat_input: dict, coords: torch.Tensor) -> sp.SparseTensor:
        opt = self.opt
        with self._loaded(stage=lambda: self._load_generator_stage("slat_generator")) as m:
            generator, embedder = m["stage"]
            generator.no_shortcut = True
            generator.inference_steps = opt["slat_inference_steps"]
            generator.rescale_t = opt["slat_rescale_t"]
            generator.reverse_fn.strength = opt["slat_cfg_strength"]
            generator.reverse_fn.interval = list(opt["slat_cfg_interval"])

            latent_shape = (slat_input["image"].shape[0], coords.shape[0], 8)
            logger.info(f"Sampling structured latent ({generator.inference_steps} steps)")
            with self._sampling_rng(), torch.autocast(device_type="cuda", dtype=self.dtype):
                cond = embedder(**slat_input)
                feats = generator(latent_shape, self.device, cond, coords.cpu().numpy())
            del m, generator, embedder
        slat = sp.SparseTensor(coords=coords, feats=feats[0]).to(self.device)
        return slat * self.slat_std.to(self.device) + self.slat_mean.to(self.device)

    @torch.no_grad()
    def decode_slat(self, slat: sp.SparseTensor, output_format: str):
        name = {"gaussian": "slat_decoder_gs", "mesh": "slat_decoder_mesh"}[output_format]
        with self._loaded(decoder=lambda: self._load_flat(name)) as m:
            logger.info(f"Decoding structured latent to {output_format}")
            with self._sampling_rng():
                return m["decoder"](slat)[0]

    # ---------------------------------------------------------------------- run
    def run(self, rgba: np.ndarray, output_format: str, seed: Optional[int] = 42, apply_pose: bool = True):
        """rgba: (H, W, 4) uint8, alpha channel = object mask.
        Returns a `Gaussian` or a `MeshExtractResult`, placed in the PyTorch3D camera frame
        (x left, y up, z forward) by the decoded pose, as the demo's `make_scene` does.
        With apply_pose=False it stays in the canonical frame (unit cube at the origin).
        The decoded pose is kept in `self.last_pose`."""
        assert output_format in FORMATS, output_format
        assert rgba.ndim == 3 and rgba.shape[-1] == 4 and rgba.dtype == np.uint8
        with self.device:
            pointmap = self.compute_pointmap(rgba)
            ss_input, moments = self._preprocess(rgba, self.ss_preprocessor, pointmap=pointmap)
            slat_input, _ = self._preprocess(rgba, self.slat_preprocessor)
            if seed is not None:
                torch.manual_seed(seed)
            self._rng = (torch.get_rng_state(), torch.cuda.get_rng_state(self.device))
            coords, pose_latents, downsample_factor = self.sample_sparse_structure(ss_input)
            # Downsampled coords span a 1/factor smaller cube; the scale compensates.
            pose = pose_utils.decode_pose(
                pose_latents, moments["pointmap_scale"], moments["pointmap_shift"], downsample_factor
            )
            self.last_pose = pose
            logger.info(
                f"Pose: scale={pose.scale[0, 0].item():.4f} translation={pose.translation[0].tolist()}"
                f" rotation={pose.rotation[0].tolist()}"
            )
            slat = self.sample_slat(slat_input, coords)
            result = self.decode_slat(slat, output_format)
            if not apply_pose:
                result.frame = "canonical"
                return result
            if output_format == "gaussian":
                pose_utils.apply_pose_gaussian(result, pose)
            else:
                pose_utils.apply_pose_mesh(result, pose)
            result.frame = "camera"
            return result
