# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch


def prune_sparse_structure(
    coord_batch,
    max_neighbor_axes_dist=1,
):
    coords, batch = coord_batch[:, 1:], coord_batch[:, 0].unsqueeze(-1)
    device = coords.device
    # 1) shift coords so minimum is zero
    min_xyz = coords.min(0)[0]
    coords0 = coords - min_xyz
    # 2) build occupancy grid
    max_xyz = coords0.max(0)[0] + 1  # size in each dim
    D, H, W = max_xyz.tolist()
    # shape (1,1,D,H,W)
    occ = torch.zeros((1, 1, D, H, W), dtype=torch.uint8, device=device)
    x, y, z = coords0.unbind(1)
    occ[0, 0, x, y, z] = 1
    # 3) 3×3×3 convolution to count each voxel + neighbors
    kernel = torch.ones(
        (
            1,
            1,
            2 * max_neighbor_axes_dist + 1,
            2 * max_neighbor_axes_dist + 1,
            2 * max_neighbor_axes_dist + 1,
        ),
        dtype=torch.uint8,
        device=device,
    )
    # pad so output is same size
    pad = max_neighbor_axes_dist
    counts = torch.nn.functional.conv3d(occ.float(), kernel.float(), padding=pad)
    # interior voxels have count == (2*max_neighbor_axes_dist+1)**3
    full_count = (2 * max_neighbor_axes_dist + 1) ** 3
    # 4) lookup counts at each original coord
    counts_at_pts = counts[0, 0, x, y, z]  # (N,)
    is_surface = counts_at_pts < full_count
    # 5) return filtered batch+coords (shift back if you want original coords)
    kept = is_surface.nonzero(as_tuple=False).squeeze(1)
    out_batch = batch[kept]
    out_coords = coords[kept]
    return torch.cat([out_batch, out_coords], dim=1)


def downsample_sparse_structure(
    coord_batch,
    max_coords=42000,
    downsample_factor=2,
):
    """
    Downsample sparse structure coordinates when there are more than max_coords.

    Downsamples by rescaling coordinates, effectively shrinking the grid while preserving
    the structure. The downsampled grid is centered in the original space.

    Args:
        coord_batch: tensor of shape (N, 4) where [:, 0] is batch index and [:, 1:] are coords
        max_coords: maximum number of coordinates to keep
            42000 should be safe number. Calculation: max(int32) / (64*768) ~= 43691
            Only needed for mesh decoding.
        downsample_factor: factor by which to downsample (e.g., 2 means half resolution)

    Returns:
        Downsampled coord_batch with coordinates rescaled if downsampling is needed
    """
    if coord_batch.shape[0] <= max_coords:
        return coord_batch, 1

    # Extract coordinates and batch indices
    coords = coord_batch[:, 1:].float()  # Shape: (N, 3), convert to float for scaling
    batch_indices = coord_batch[:, 0:1]  # Shape: (N, 1)

    # Find the actual coordinate bounds
    coords_min = coords.min(dim=0)[0]  # Shape: (3,)
    coords_max = coords.max(dim=0)[0]  # Shape: (3,)
    original_size = coords_max - coords_min + 1  # Add 1 since coordinates are discrete

    # Calculate target size after downsampling
    target_size = original_size / downsample_factor

    # Calculate the offset to center the downsampled grid
    offset = (original_size - target_size) / 2
    target_min = coords_min + offset
    target_max = coords_min + offset + target_size - 1

    # Normalize coordinates to [0, 1] within their actual range
    coords_normalized = (coords - coords_min) / (coords_max - coords_min)

    # Scale to the target range
    coords_rescaled = coords_normalized * (target_size - 1) + target_min

    # Round to integers to get discrete grid coordinates
    coords_rescaled = torch.round(coords_rescaled).int()

    # Clamp to ensure we stay within bounds
    coords_rescaled = torch.clamp(coords_rescaled, target_min.int(), target_max.int())

    # Remove duplicates that may have been created by the downsampling
    # Concatenate batch and coords for duplicate removal
    combined = torch.cat([batch_indices, coords_rescaled], dim=1)
    unique_combined = torch.unique(combined, dim=0)

    # If still too many after deduplication, randomly subsample
    if unique_combined.shape[0] > max_coords:
        indices = torch.randperm(unique_combined.shape[0], device=coord_batch.device)[
            :max_coords
        ]
        unique_combined = unique_combined[indices]

    return unique_combined.int(), downsample_factor
