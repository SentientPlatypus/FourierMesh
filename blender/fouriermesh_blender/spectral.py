"""Dense graph Laplacian spectral core (numpy only, no scipy)."""

from __future__ import annotations

import numpy as np


def combinatorial_laplacian_from_edges(
    edges: np.ndarray, n: int, *, normalized: bool = False
) -> np.ndarray:
    """Dense graph Laplacian ``L = D - W`` from an ``(E, 2)`` edge array."""
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
    """Eigendecomposition of the mesh Laplacian.

    Returns ``(U, coeffs, lambdas)`` with ``U`` of shape ``(N, N)``,
    ``coeffs = U.T @ V``, and ascending ``lambdas``.
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
    """Reconstruct vertices from the first ``k`` modes."""
    k = max(1, min(int(k), U.shape[1]))
    return U[:, :k] @ coeffs[:k]
