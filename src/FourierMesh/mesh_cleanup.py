"""Basic mesh cleanup helpers for STL export."""

from __future__ import annotations

import numpy as np


def remove_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    rel_eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop triangles with near-zero area."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if f.shape[0] == 0:
        return v, f
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    scale = float(np.max(v.max(axis=0) - v.min(axis=0)) + 1e-30)
    thresh = rel_eps * (scale**2)
    return v, f[areas > thresh]


def remove_duplicate_faces(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop faces that reuse the same three vertex indices (any winding)."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if f.shape[0] == 0:
        return v, f
    keys = np.sort(f, axis=1)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return v, f[np.sort(idx)]


def compact_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remap vertices to 0..n-1 after faces were removed."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if f.size == 0:
        return v[:0], f
    used = np.unique(f.ravel())
    v_new = v[used]
    inv = -np.ones(v.shape[0], dtype=np.int64)
    inv[used] = np.arange(used.size, dtype=np.int64)
    return v_new, inv[f]
