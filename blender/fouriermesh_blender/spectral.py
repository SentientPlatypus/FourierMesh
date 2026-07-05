"""Numpy-only graph-Laplacian spectral core for the FourierMesh Blender add-on.

A dependency-light subset of the main FourierMesh library (the dense
``numpy.linalg.eigh`` path only), so the add-on runs against Blender's bundled
numpy with no scipy install or wheel bundling required.

The Laplacian is built directly from an edge list, which Blender's mesh data
provides natively (``mesh.edges``) -- no triangulation needed, and n-gons /
quads work as-is.
"""

from __future__ import annotations

import numpy as np


def combinatorial_laplacian_from_edges(
    edges: np.ndarray, n: int, *, normalized: bool = False
) -> np.ndarray:
    """Dense graph Laplacian from an ``(E, 2)`` edge array.

    ``L = D - W`` (combinatorial) or ``I - D^-1/2 W D^-1/2`` (normalized).
    """
    edges = np.asarray(edges, dtype=np.int64)
    W = np.zeros((n, n), dtype=np.float64)
    if edges.size:
        i = edges[:, 0]
        j = edges[:, 1]
        W[i, j] = 1.0
        W[j, i] = 1.0
    degrees = W.sum(axis=1)
    if normalized:
        inv_sqrt = np.zeros_like(degrees)
        nonzero = degrees > 0
        inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        D_inv_sqrt = np.diag(inv_sqrt)
        return np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    return np.diag(degrees) - W


def solve_eigenbasis(
    vertices: np.ndarray, edges: np.ndarray, *, normalized: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full dense eigendecomposition of the mesh graph Laplacian.

    This is the expensive step -- run it once and cache the result. Returns
    ``(U, coeffs, lambdas)`` where ``U`` is ``(N, N)`` (eigenvectors as columns,
    ascending eigenvalue), ``coeffs = U.T @ V`` is ``(N, 3)``, and ``lambdas``
    is ``(N,)``. Reconstruct any ``k`` cheaply via :func:`reconstruct`.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    n = int(vertices.shape[0])
    L = combinatorial_laplacian_from_edges(edges, n, normalized=normalized)
    lambdas, U = np.linalg.eigh(L)
    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    U = U[:, order]
    coeffs = U.T @ vertices
    return U, coeffs, lambdas


def reconstruct(U: np.ndarray, coeffs: np.ndarray, k: int) -> np.ndarray:
    """Low-pass reconstruction from the first ``k`` modes -- a truncated matmul.

    Cheap enough to call on every redo-panel drag: ``k`` is a free runtime knob
    once the basis in ``U``/``coeffs`` has been solved.
    """
    k = max(1, min(int(k), U.shape[1]))
    return U[:, :k] @ coeffs[:k]
