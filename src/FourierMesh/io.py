"""STL mesh I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stl import mesh


def load_mesh_stl(
    filename: str | Path,
    *,
    dedupe: bool = True,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an STL file as deduplicated vertices and triangle faces.

    Returns ``(vertices, faces)`` with shapes ``(N, 3)`` and ``(M, 3)``.
    """
    stl_mesh = mesh.Mesh.from_file(str(filename))
    raw_vertices = stl_mesh.vectors.reshape(-1, 3)

    if not dedupe:
        vertices = raw_vertices
        faces = np.arange(len(raw_vertices), dtype=np.int64).reshape(-1, 3)
        return vertices.astype(float), faces

    rounded = np.round(raw_vertices, decimals=decimals)
    vertices, inverse = np.unique(rounded, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int64)
    return vertices.astype(float), faces


def load_stl_as_vertices_faces(
    filename: str | Path,
    dedupe: bool = True,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Alias for :func:`load_mesh_stl` (backward compatible name)."""
    return load_mesh_stl(filename, dedupe=dedupe, decimals=decimals)


def save_mesh_stl(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: str | Path,
    *,
    fmt: str = "binary",
    name: str = "mesh",
) -> None:
    """Write triangle soup as STL via ``numpy-stl`` (binary or ASCII)."""
    from stl import Mesh, Mode
    from stl.base import RemoveDuplicates

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    n_tri = int(f.shape[0])
    if n_tri == 0:
        raise ValueError("Cannot write STL: no triangles")
    if f.min() < 0 or f.max() >= v.shape[0]:
        raise ValueError("face indices out of bounds for vertex array")
    if not np.isfinite(v).all():
        raise ValueError("Cannot write STL: non-finite vertex coordinates")

    stl_data = np.zeros(n_tri, dtype=Mesh.dtype)
    stl_data["vectors"] = v[f]
    m = Mesh(
        stl_data,
        calculate_normals=True,
        remove_empty_areas=False,
        remove_duplicate_polygons=RemoveDuplicates.SINGLE,
        name=name,
    )
    mode = Mode.ASCII if fmt == "ascii" else Mode.BINARY
    m.save(str(path), mode=mode, update_normals=True)
