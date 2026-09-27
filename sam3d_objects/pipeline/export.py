"""Write pipeline outputs to disk."""

from pathlib import Path

import numpy as np
import trimesh

# The decoders work in a z-up frame; glTF is y-up (same rotation as the released to_glb).
_Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


def save_gaussian(gaussian, path: Path) -> Path:
    """Standard 3DGS .ply (loadable by any 3DGS viewer)."""
    gaussian.save_ply(str(path))
    return path


def save_mesh(mesh, path: Path) -> Path:
    """Vertex-coloured .glb, i.e. what the official demo produces
    (to_glb(with_mesh_postprocess=False, with_texture_baking=False, use_vertex_color=True))."""
    vertices = mesh.vertices.float().cpu().numpy() @ _Z_UP_TO_Y_UP
    faces = mesh.faces.cpu().numpy()
    out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    out.visual.vertex_colors = mesh.vertex_attrs[:, :3].float().cpu().numpy()
    out.export(str(path))
    return path
