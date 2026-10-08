"""Stage-1 pose tokens -> object-to-camera similarity transform, and applying it.

Port of the released `inference_utils.pose_decoder("ScaleShiftInvariant")` (the
convention `pipeline.yaml::pose_decoder_name` selects) and of the demo's
`notebook/inference.py::make_scene`, without the pytorch3d dependency.

Conventions are pytorch3d's: quaternions are (w, x, y, z), transforms act on row
vectors, so a local point x maps to the camera frame as `(x * scale) @ R + t`
with R = quaternion_to_matrix(rotation). The camera frame is PyTorch3D's
(x left, y up, z forward).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Moments the 6D rotation token was normalised with (inference_utils.ROTATION_6D_*).
_ROTATION_6D_MEAN = torch.tensor([
    -0.06366084883674913, 0.008438224692279752, 0.00017084786438302483,
    0.0007126610473540038, -0.0030916726538816417, 0.5166093753457688,
])
_ROTATION_6D_STD = torch.tensor([
    0.6656971967514863, 0.6787012271867754, 0.30345010594844524,
    0.4394504420678794, 0.39817973931717104, 0.6176286868761914,
])


@dataclass
class Pose:
    rotation: torch.Tensor  # (1, 4) quaternion wxyz
    translation: torch.Tensor  # (1, 3)
    scale: torch.Tensor  # (1, 3), isotropic


# ------------------------------------------------------- pytorch3d.transforms
def _standardize_quaternion(q: torch.Tensor) -> torch.Tensor:
    return torch.where(q[..., 0:1] < 0, -q, q)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = torch.stack(
        [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1
    )
    q_abs = torch.where(q_abs > 0, q_abs, torch.zeros_like(q_abs)).sqrt()
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp_min(0.1))
    best = F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5
    return _standardize_quaternion(quat_candidates[best, :].reshape(batch_dim + (4,)))


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    out = torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )
    return _standardize_quaternion(out)


def quaternion_invert(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1, -1, -1, -1])


# ------------------------------------------------------------------- decoding
def _rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    b1 = F.normalize(rot_6d[..., 0:3], dim=-1)
    a2 = rot_6d[..., 3:6]
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def decode_pose(latents: dict, scene_scale: torch.Tensor, scene_shift: torch.Tensor, downsample_factor=1) -> Pose:
    """latents: stage-1 generator outputs ('6drotation_normalized', 'scale', 'translation'),
    each (1, 1, C). scene_scale / scene_shift: (1, 3) pointmap normalisation moments.
    downsample_factor: what `downsample_sparse_structure` shrank the voxel coords by."""
    rot_6d = latents["6drotation_normalized"].float()
    rot_6d = rot_6d * _ROTATION_6D_STD.to(rot_6d) + _ROTATION_6D_MEAN.to(rot_6d)
    rotation = matrix_to_quaternion(_rotation_6d_to_matrix(rot_6d))
    scale = latents["scale"].float().exp()
    translation = latents["translation"].float()

    # ScaleShiftInvariant.to_instance_pose (normalize=False): post-compose the pose
    # predicted in the normalised pointmap space with ssi_to_metric = scale(S).translate(shift),
    # then decompose the result back into scale / rotation / translation.
    S = scene_scale.float().reshape(-1, 1, 3).to(scale)
    shift = scene_shift.float().reshape(-1, 1, 3).to(scale)
    linear = scale.unsqueeze(-1) * quaternion_to_matrix(rotation) * S.unsqueeze(-2)  # diag(s) R diag(S)
    scale = linear.norm(dim=-1)
    rotation = matrix_to_quaternion(linear / scale.unsqueeze(-1))
    translation = translation * S + shift

    scale = scale.squeeze(0).mean(-1, keepdim=True).expand(1, 3) * downsample_factor
    return Pose(rotation=rotation.squeeze(0), translation=translation.squeeze(0), scale=scale)


# ------------------------------------------------------------------- applying
def _to_camera(points: torch.Tensor, pose: Pose) -> torch.Tensor:
    R = quaternion_to_matrix(pose.rotation[0]).to(points)
    return (points * pose.scale[0].to(points)) @ R + pose.translation[0].to(points)


def apply_pose_gaussian(gaussian, pose: Pose):
    """In place, as `make_scene` does for a single object."""
    s = pose.scale[0, 0].item()
    xyz = _to_camera(gaussian.get_xyz, pose)
    gaussian.from_xyz(xyz)
    rots = quaternion_multiply(quaternion_invert(pose.rotation).to(xyz), gaussian.get_rotation)
    gaussian.from_rotation(rots)
    scaling = gaussian.get_scaling * s
    gaussian.mininum_kernel_size *= s
    gaussian.from_scaling(torch.clamp_min(scaling, gaussian.mininum_kernel_size * 1.1))
    return gaussian


def apply_pose_mesh(mesh, pose: Pose):
    """In place: vertices into the camera frame (the demo has no mesh counterpart of
    make_scene; this is the same transform applied to the mesh vertices)."""
    mesh.vertices = _to_camera(mesh.vertices.float(), pose)
    mesh.face_normal = mesh.comput_face_normals(mesh.vertices, mesh.faces)
    return mesh
