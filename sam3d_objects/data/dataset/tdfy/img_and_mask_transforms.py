# Copyright (c) Meta Platforms, Inc. and affiliates.
from collections import namedtuple
from typing import Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from loguru import logger


class BoundingBoxError(Exception):
    pass


def check_bounding_box(bbox_w, bbox_h):
    if bbox_w < 2 or bbox_h < 2:
        raise BoundingBoxError("Bounding box dimensions must be at least 2x2.")


def concat_rgba(
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Create a 4-channel RGBA image from a 3-channel RGB image and a mask.
    """
    assert rgb_image.dim() == 3, f"{rgb_image.shape=}"
    assert mask.dim() == 2, f"{mask.shape=}"
    assert rgb_image.shape[0] == 3, f"{rgb_image.shape[0]=}"
    assert rgb_image.shape[1:] == mask.shape, f"{rgb_image.shape[1:]=} != {mask.shape=}"
    return torch.cat((rgb_image, mask[None, ...]), dim=0)


def split_rgba(rgba_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split a 4-channel RGBA image into a 3-channel RGB image and a 1-channel mask.

    Args:
        rgba_image: A 4-channel RGBA image.

    Returns:
        A tuple of (rgb_image, mask).
    """
    assert rgba_image.dim() == 3, f"{rgba_image.shape=}"
    assert rgba_image.shape[0] == 4, f"{rgba_image.shape[0]=}"
    return rgba_image[:3], rgba_image[3]


def get_mask(
    rgb_image: torch.Tensor,
    depth_image: torch.Tensor,
    mask_source: str,
) -> torch.Tensor:
    """
    Extract a mask from either the alpha channel of an RGB image or a depth image.

    Args:
        rgb_image: Tensor of shape (B, C, H, W) or (C, H, W) where C >= 4 if using alpha channel
        depth_image: Tensor of shape (B, 1, H, W) or (1, H, W) containing depth information
        mask_source: Source of the mask, either "ALPHA_CHANNEL" or "DEPTH"

    Returns:
        mask: Tensor of shape (B, 1, H, W) or (1, H, W) containing the extracted mask
    """
    # Handle unbatched inputs (add batch dimension if needed)
    is_batched = len(rgb_image.shape) == 4

    if not is_batched:
        rgb_image = rgb_image.unsqueeze(0)
        if depth_image is not None:
            depth_image = depth_image.unsqueeze(0)

    if mask_source == "ALPHA_CHANNEL":
        if rgb_image.shape[1] != 4:
            logger.warning(f"No ALPHA CHANNEL for the image, cannot read mask.")
            mask = None
        else:
            mask = rgb_image[:, 3:4, :, :]
    elif mask_source == "DEPTH":
        mask = depth_image
    else:
        raise ValueError(f"Invalid mask source: {mask_source}")

    # Remove batch dimension if input was unbatched
    if not is_batched:
        mask = mask.squeeze(0)

    return mask


def crop_around_mask_with_padding(
    loaded_image: torch.Tensor,
    mask: torch.Tensor,
    box_size_factor: float = 1.6,
    padding_factor: float = 0.1,
    pointmap: Optional[torch.Tensor] = None,
) -> np.ndarray:
    # cast to ensure the function can be called normally
    cast_mask = False
    if mask.dim() == 3:
        assert mask.shape[0] == 1, "cannot take mask with channel dimension not 1"
        mask = mask[0]
        cast_mask = True
    loaded_image = concat_rgba(loaded_image, mask)

    bbox = compute_mask_bbox(mask, box_size_factor)
    loaded_image = torchvision.transforms.functional.crop(
        loaded_image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]
    )

    # Crop pointmap if provided
    if pointmap is not None:
        pointmap = torchvision.transforms.functional.crop(
            pointmap, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]
        )

    C, H, W = loaded_image.shape
    max_dim = max(H, W)  # Get the larger dimension

    # Step 1: Pad to square shape
    pad_h = (max_dim - H) // 2
    pad_w = (max_dim - W) // 2
    pad_h_extra = (max_dim - H) - pad_h  # To ensure even padding
    pad_w_extra = (max_dim - W) - pad_w

    loaded_image = torch.nn.functional.pad(
        loaded_image, (pad_w, pad_w_extra, pad_h, pad_h_extra), mode="constant", value=0
    )
    if pointmap is not None:
        pointmap = torch.nn.functional.pad(
            pointmap,
            (pad_w, pad_w_extra, pad_h, pad_h_extra),
            mode="constant",
            value=float("nan"),
        )

    # Step 2: Extend by 10% on each side; idk but this seems to have better results overall
    if padding_factor > 0:
        extend_size = int(max_dim * padding_factor)  # 10% extension on each side
        loaded_image = torch.nn.functional.pad(
            loaded_image,
            (extend_size, extend_size, extend_size, extend_size),
            mode="constant",
            value=0,
        )

        if pointmap is not None:
            pointmap = torch.nn.functional.pad(
                pointmap,
                (extend_size, extend_size, extend_size, extend_size),
                mode="constant",
                value=float("nan"),
            )

    rgb_image, mask = split_rgba(loaded_image)
    if cast_mask:
        mask = mask[None]

    if pointmap is not None:
        return rgb_image, mask, pointmap
    return rgb_image, mask


def compute_mask_bbox(
    mask: torch.Tensor, box_size_factor: float = 1.0
) -> tuple[float, float, float, float]:
    """
    Compute a bounding box around a binary mask with optional size adjustment.

    Args:
        mask: A 2D binary tensor where non-zero values represent the object of interest.
        box_size_factor: Factor to scale the bounding box size. Values > 1.0 create a larger box.
            Default is 1.0 (tight bounding box).

    Returns:
        A tuple of (x1, y1, x2, y2) coordinates representing the bounding box,
        where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Raises:
        ValueError: If mask is not a torch.Tensor or not a 2D tensor.
    """
    if not isinstance(mask, torch.Tensor):
        raise ValueError("Mask must be a torch.Tensor")
    if not mask.dim() == 2:
        raise ValueError("Mask must be a 2D tensor")
    bbox_indices = torch.nonzero(mask)
    if bbox_indices.numel() == 0:
        # Handle empty mask case
        return (0, 0, 0, 0)

    y_indices = bbox_indices[:, 0]
    x_indices = bbox_indices[:, 1]

    min_x = torch.min(x_indices).item()
    min_y = torch.min(y_indices).item()
    max_x = torch.max(x_indices).item()
    max_y = torch.max(y_indices).item()

    bbox = (min_x, min_y, max_x, max_y)

    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    check_bounding_box(bbox_w, bbox_h)

    size = max(bbox_w, bbox_h, 2)
    size = int(size * box_size_factor)

    bbox = (
        int(center_x - size // 2),
        int(center_y - size // 2),
        int(center_x + size // 2),
        int(center_y + size // 2),
    )
    # bbox = tuple(map(int, bbox))
    return bbox


def crop_and_pad(image, bbox):
    """
    Crop an image using a bounding box and pad with zeros if out of bounds.

    Args:
        image (torch.Tensor): CxHxW image.
        bbox (tuple): (x1, y1, x2, y2) bounding box.

    Returns:
        torch.Tensor: Cropped and zero-padded image.
    """
    C, H, W = image.shape
    x1, y1, x2, y2 = bbox

    # Ensure coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Compute cropping coordinates
    x1_pad, y1_pad = max(0, -x1), max(0, -y1)
    x2_pad, y2_pad = max(0, x2 - W), max(0, y2 - H)

    # Compute valid region in the original image
    x1_crop, y1_crop = max(0, x1), max(0, y1)
    x2_crop, y2_crop = min(W, x2), min(H, y2)

    # Extract the valid part
    cropped = image[:, y1_crop:y2_crop, x1_crop:x2_crop]

    # Create a zero-padded output
    padded = torch.zeros((C, y2 - y1, x2 - x1), dtype=image.dtype)

    # Place the cropped image into the zero-padded array
    padded[
        :, y1_pad : y1_pad + cropped.shape[1], x1_pad : x1_pad + cropped.shape[2]
    ] = cropped

    return padded


def resize_all_to_same_size(
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
    pointmap: Optional[torch.Tensor] = None,
    target_size: Optional[tuple[int, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Resize RGB image, mask, and pointmap to the same size.
    
    This is crucial when pointmaps have different resolution than RGB images,
    which must be done BEFORE any cropping operations.
    
    Args:
        rgb_image: RGB image tensor of shape (C, H, W)
        mask: Mask tensor of shape (H, W) or (1, H, W)
        pointmap: Optional pointmap tensor of shape (C_p, H_p, W_p)
        target_size: Target size as (H, W). If None, uses RGB image size.
        
    Returns:
        Tuple of (resized_rgb, resized_mask, resized_pointmap)
    """
    squeeze_mask = (mask.dim() == 2) 
    if squeeze_mask:
        mask = mask.unsqueeze(0)
    
    if target_size is None:
        target_size = (rgb_image.shape[1], rgb_image.shape[2])  # (H, W)
    
    rgb_needs_resize = (rgb_image.shape[1], rgb_image.shape[2]) != target_size
    if rgb_needs_resize:
        rgb_image = torchvision.transforms.functional.resize(
            rgb_image, target_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        mask = torchvision.transforms.functional.resize(
            mask, target_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
    
    if pointmap is not None:
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        if pointmap_size != target_size:
            # Handle NaN values in pointmap during resizing
            # Direct resize would propagate NaN values, so we need special handling
            nan_mask = torch.isnan(pointmap).any(dim=0)
            pointmap_clean = torch.where(torch.isnan(pointmap), torch.zeros_like(pointmap), pointmap)
            pointmap_resized = torchvision.transforms.functional.resize(
                pointmap_clean, target_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            
            # Resize the nan mask to identify which regions should remain invalid
            nan_mask_resized = torchvision.transforms.functional.resize(
                nan_mask.unsqueeze(0).float(), target_size, 
                interpolation=torchvision.transforms.InterpolationMode.NEAREST
            ).squeeze(0) > 0.5
            
            # Restore NaN values in regions that were originally invalid
            pointmap = torch.where(
                nan_mask_resized.unsqueeze(0).expand_as(pointmap_resized),
                torch.full_like(pointmap_resized, float('nan')),
                pointmap_resized
            )
    
    if squeeze_mask:
        mask = mask.squeeze(0)
    
    if pointmap is not None:
        return rgb_image, mask, pointmap
    return rgb_image, mask


SSINormalizedPointmap = namedtuple("SSINormalizedPointmap", ["pointmap", "scale", "shift"])
class SSIPointmapNormalizer:
    """Interface: normalize(pointmap, mask, scale=None, shift=None) -> SSINormalizedPointmap."""

    def normalize(self, pointmap, mask, scale=None, shift=None) -> SSINormalizedPointmap:
        raise NotImplementedError


class ObjectCentricSSI(SSIPointmapNormalizer):
    def __init__(self,
        use_scene_scale: bool = True,
        quantile_drop_threshold: float = 0.1,
        clip_beyond_scale: Optional[float] = None,
        # scale_factor: float = 3.8076, # e^(1.337); empirical mean of R3+Artist train
        scale_factor: float = 1.0, # e^(1.337); empirical mean of R3+Artist train
        allow_scale_and_shift_override: bool = False,
        raise_on_no_valid_points: bool = False,
    ):
        self.use_scene_scale = use_scene_scale
        self.quantile_drop_threshold = quantile_drop_threshold
        self.clip_beyond_scale = clip_beyond_scale
        self.scale_factor = scale_factor
        self.allow_scale_and_shift_override = allow_scale_and_shift_override
        self.raise_on_no_valid_points = raise_on_no_valid_points

    def _compute_scale_and_shift(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])

        
        mask_resized = torchvision.transforms.functional.resize(
            mask, pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        ).squeeze(0)

        pointmap_flat = pointmap.reshape(3, -1)
        # Get valid points from the mask
        mask_bool = mask_resized.reshape(-1) > 0.5
        mask_points = pointmap_flat[:, mask_bool]

        if mask_points.isfinite().max() == 0:
            if self.raise_on_no_valid_points:
                raise ValueError(f"No valid points found in mask")
            logger.warning(f"No valid points found in mask; setting scale to {self.scale_factor} and shift to 0")
            return torch.ones_like(pointmap_flat[:,0]) * self.scale_factor, torch.zeros_like(pointmap_flat[:,0])

        # Compute median for shift
        shift = mask_points.nanmedian(dim=-1).values
        # logger.info(f"{pointmap.shape=} {mask_resized.shape=} {shift.shape=}")


        if self.use_scene_scale == True:
            # Normalize by the scene scale
            points_centered = pointmap_flat - shift.unsqueeze(-1)
            max_dims = points_centered.abs().max(dim=0).values
            scale = max_dims.nanmedian(dim=-1).values
        elif self.use_scene_scale == False:
            # Normalize by the object scale
            shifted_mask_points = mask_points - shift.unsqueeze(-1)
            norm = shifted_mask_points.norm(dim=0)
            quantiles = torch.nanquantile(norm,
                torch.tensor([self.quantile_drop_threshold, 1. - self.quantile_drop_threshold],
                device=shifted_mask_points.device),
                dim=-1)
            scale = (quantiles[1] - quantiles[0]).max(dim=-1).values * 2.0
        elif self.use_scene_scale.upper() == "OBJECT_NORM_MEDIAN":
            # Normalize by the object scale
            shifted_mask_points = mask_points - shift.unsqueeze(-1)
            norm = shifted_mask_points.norm(dim=0)
            scale = norm.nanmedian(dim=-1).values
        else:
            raise ValueError(f"Invalid use_scene_scale: {self.use_scene_scale}")
        scale = scale.expand_as(shift) # per-dim scaling
        scale = scale * self.scale_factor
        return scale, shift
    
    def normalize(self, pointmap: torch.Tensor, mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None, shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. resize mask to size of pointmap using nearest interpolation
        # 2. get mask points: pointmap[mask > 0.5]
        # 3. shift = mask_points.median() # xyz
        # 4. scale = # filter. If no points, then
        # logger.info(f"{pointmap.shape=} {mask.shape=}")
        assert pointmap.shape[0] == 3, "pointmap must be in (3, H, W) format"
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])

        _scale, _shift = self._compute_scale_and_shift(pointmap, mask)
        if scale is not None and self.allow_scale_and_shift_override:
            _scale = scale
        if shift is not None and self.allow_scale_and_shift_override:
            _shift = shift
        return_scale, return_shift = _scale, _shift

        # Apply normalization
        pointmap_normalized = _apply_metric_to_ssi(pointmap, return_scale, return_shift)
        
        if self.clip_beyond_scale is not None and self.clip_beyond_scale > 0:
            new_norm = pointmap_normalized.norm(dim=0)
            pointmap_normalized = torch.where(
                new_norm > self.clip_beyond_scale,
                torch.full_like(pointmap_normalized, float('nan')),
                pointmap_normalized
            )

        return SSINormalizedPointmap(pointmap_normalized, return_scale, return_shift)


def _apply_metric_to_ssi(pointmap: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Metric -> scale/shift-invariant space: (p - shift) / scale, for a (3, H, W) pointmap.

    Written as the same homogeneous 4x4 products PyTorch3D's
    `Transform3d().scale(scale).translate(shift).inverse().transform_points(...)`
    performs (row-vector convention), so results -- including how non-finite
    points turn into NaN -- match the released pipeline without needing pytorch3d.
    """
    assert pointmap.shape[0] == 3, "pointmap must be in (3, H, W) format"
    scale, shift = scale.reshape(3), shift.reshape(3)
    inv_translate = torch.eye(4, dtype=shift.dtype, device=shift.device)
    inv_translate[3, :3] = -shift
    inv_scale = torch.diag(torch.cat([1.0 / scale, scale.new_ones(1)])).to(shift.device)
    matrix = torch.eye(4, dtype=shift.dtype, device=shift.device)[None].bmm(inv_translate[None]).bmm(inv_scale[None])

    points = pointmap.permute(1, 2, 0).reshape(1, -1, 3)
    points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    points = points.bmm(matrix)
    points = points[..., :3] / points[..., 3:]
    return points.reshape(pointmap.shape[1], pointmap.shape[2], 3).permute(2, 0, 1)
