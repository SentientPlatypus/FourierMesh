"""Mesh Laplacian + graph Fourier round-trip on David STL (submesh for speed)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from FourierMesh import (
    inverse_mesh_fourier,
    load_mesh_stl,
    mesh_fourier_laplacian,
)


def _david_stl_path() -> Path:
    env = os.environ.get("FOURIERMESH_DAVID_STL")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent / "fixtures" / "DavidStatue.stl"


def _submesh(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """First ``max_faces`` triangles, vertices remapped to 0..n-1."""
    faces = np.asarray(faces, dtype=int)
    if faces.shape[0] > max_faces:
        faces = faces[:max_faces].copy()
    used = np.unique(faces.ravel())
    v_sub = np.asarray(vertices, dtype=float)[used]
    old_to_new = -np.ones(vertices.shape[0], dtype=np.int64)
    old_to_new[used] = np.arange(used.size, dtype=np.int64)
    f_sub = old_to_new[faces]
    return v_sub, f_sub


@pytest.mark.skipif(
    not _david_stl_path().is_file(),
    reason=(
        "David STL not found. Place DavidStatue.stl under tests/fixtures/ "
        "or set FOURIERMESH_DAVID_STL to the file path."
    ),
)
def test_mesh_laplacian_inverse_fourier_round_trip_on_david_submesh():
    """
    Load David STL, build combinatorial graph Laplacian on a small face-induced
    submesh, graph-Fourier transform vertex positions, and invert to recover
    geometry. Full mesh eigendecomposition is O(N^3); we keep N modest.
    """
    stl_path = _david_stl_path()
    vertices, faces = load_mesh_stl(str(stl_path), dedupe=True)

    max_faces = 80
    v_sub, f_sub = _submesh(vertices, faces, max_faces=max_faces)
    n = v_sub.shape[0]
    assert n >= 4
    assert f_sub.shape[0] <= max_faces

    coeffs, U, lambdas, L = mesh_fourier_laplacian(v_sub, faces=f_sub, normalized=False)

    assert coeffs.shape == (n, 3)
    assert U.shape == (n, n)
    assert lambdas.shape == (n,)
    assert L.shape == (n, n)
    assert np.allclose(L, L.T, atol=1e-12, rtol=0.0)

    # Constant vector on a connected component: eigenvalue 0 (numerically small).
    assert lambdas[0] <= 1e-8 * max(1.0, float(np.max(np.abs(lambdas))))

    recon_full = inverse_mesh_fourier(coeffs, U, k=None)
    assert np.allclose(recon_full, v_sub, atol=1e-9, rtol=1e-9)

    # Single-mode reconstruction loses spatial detail vs full spectrum.
    recon_one_mode = inverse_mesh_fourier(coeffs, U, k=1)
    assert not np.allclose(recon_one_mode, v_sub, atol=1e-4, rtol=1e-4)
