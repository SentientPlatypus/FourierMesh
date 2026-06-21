"""Unit tests for graph Laplacian mesh Fourier APIs."""

from __future__ import annotations

import numpy as np
import pytest

from FourierMesh import (
    inverse_mesh_fourier,
    mesh_fourier_laplacian,
    mesh_fourier_transform,
    mesh_laplacian_eigenmodes,
    reconstruct_mesh,
)
from FourierMesh.laplacian import _DENSE_FULL_THRESHOLD


def _unit_cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return v, f


def test_mesh_fourier_laplacian_full_round_trip_on_cube():
    v, f = _unit_cube_mesh()
    n = v.shape[0]
    coeffs, U, lambdas, L = mesh_fourier_laplacian(v, faces=f, normalized=False)

    assert coeffs.shape == (n, 3)
    assert U.shape == (n, n)
    assert lambdas.shape == (n,)
    assert L.shape == (n, n)
    assert np.allclose(L, L.T, atol=1e-12, rtol=0.0)
    assert lambdas[0] <= 1e-8 * max(1.0, float(np.max(np.abs(lambdas))))

    recon = inverse_mesh_fourier(coeffs, U, k=None)
    assert np.allclose(recon, v, atol=1e-9, rtol=1e-9)


def test_reconstruct_mesh_lowpass_differs_from_original():
    v, f = _unit_cube_mesh()
    v_smooth, lambdas = reconstruct_mesh(v, f, k=1)
    assert lambdas.shape == (1,)
    assert not np.allclose(v_smooth, v, atol=1e-4, rtol=1e-4)


def test_reconstruct_mesh_full_spectrum_matches_original():
    v, f = _unit_cube_mesh()
    n = v.shape[0]
    v_recon, lambdas = reconstruct_mesh(v, f, k=n)
    assert np.allclose(v_recon, v, atol=1e-9, rtol=1e-9)


def test_mesh_fourier_transform_matches_manual_projection():
    v, f = _unit_cube_mesh()
    coeffs, U, lambdas = mesh_fourier_transform(v, f, k=4)
    assert coeffs.shape == (4, 3)
    assert U.shape == (v.shape[0], 4)
    assert lambdas.shape == (4,)
    assert np.allclose(coeffs, U.T @ v, atol=1e-10, rtol=0.0)


def test_mesh_laplacian_eigenmodes_sparse_path_on_large_grid():
    """Synthetic mesh large enough to use sparse eigsh."""
    nx, ny = 50, 50
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    xv, yv = np.meshgrid(xs, ys)
    zv = np.zeros_like(xv)
    vertices = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
    faces = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            a = i * nx + j
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces = np.asarray(faces, dtype=np.int64)
    assert vertices.shape[0] > _DENSE_FULL_THRESHOLD

    lambdas, U = mesh_laplacian_eigenmodes(vertices, faces, k=8)
    assert lambdas.shape == (8,)
    assert U.shape == (vertices.shape[0], 8)
    assert lambdas[0] <= 1e-6


def test_mesh_fourier_laplacian_requires_faces_or_edges():
    v, _ = _unit_cube_mesh()
    with pytest.raises(ValueError, match="faces or edges"):
        mesh_fourier_laplacian(v)
