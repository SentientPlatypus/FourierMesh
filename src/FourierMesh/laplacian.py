"""Graph Laplacian spectral analysis and reconstruction for triangle meshes."""

from __future__ import annotations

import numpy as np

_DENSE_FULL_THRESHOLD = 2048


def _build_adjacency_from_faces(faces: np.ndarray, n: int) -> np.ndarray:
    """Dense weighted adjacency matrix W from triangle faces."""
    faces = np.asarray(faces, dtype=np.int64)
    W = np.zeros((n, n), dtype=float)
    for tri in faces:
        i, j, k = (int(x) for x in tri)
        W[i, j] = W[j, i] = 1.0
        W[j, k] = W[k, j] = 1.0
        W[k, i] = W[i, k] = 1.0
    return W


def _combinatorial_laplacian(W: np.ndarray, *, normalized: bool) -> np.ndarray:
    degrees = np.sum(W, axis=1)
    if normalized:
        inv_sqrt_deg = np.zeros_like(degrees)
        nonzero = degrees > 0
        inv_sqrt_deg[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        D_inv_sqrt = np.diag(inv_sqrt_deg)
        return np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return np.diag(degrees) - W


def _sparse_laplacian_from_faces(
    faces: np.ndarray, n: int, *, normalized: bool
):
    from scipy import sparse
    from scipy.sparse import csgraph

    faces = np.asarray(faces, dtype=np.int64)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    data = np.ones(2 * edges.shape[0], dtype=float)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    W = sparse.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    return csgraph.laplacian(W, normed=normalized).astype(float)


def mesh_fourier_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray | None = None,
    edges: np.ndarray | None = None,
    normalized: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full dense graph Laplacian eigendecomposition (small meshes only).

    For large meshes or partial spectra, use :func:`mesh_laplacian_eigenmodes`
    or :func:`reconstruct_mesh`.

    Returns ``(coeffs, U, lambdas, L)``.
    """
    vertices = np.asarray(vertices, dtype=float)
    n = int(vertices.shape[0])
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")

    if faces is not None:
        faces = np.asarray(faces, dtype=np.int64)
        W = _build_adjacency_from_faces(faces, n)
    elif edges is not None:
        edges = np.asarray(edges, dtype=np.int64)
        W = np.zeros((n, n), dtype=float)
        for i, j in edges:
            W[int(i), int(j)] = 1.0
            W[int(j), int(i)] = 1.0
    else:
        raise ValueError("Pass either faces or edges")

    L = _combinatorial_laplacian(W, normalized=normalized)
    lambdas, U = np.linalg.eigh(L)
    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    U = U[:, order]
    coeffs = U.T @ vertices
    return coeffs, U, lambdas, L


def mesh_laplacian_eigenmodes(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    k: int | None = None,
    normalized: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lowest graph-Laplacian eigenpairs for a triangle mesh.

    Uses dense ``eigh`` when ``N <= 2048`` and a full spectrum is requested;
    otherwise uses ``scipy.sparse.linalg.eigsh``.

    Returns ``(lambdas, U)`` with ``U`` of shape ``(N, k_eff)``.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    n = int(vertices.shape[0])
    if n == 0:
        raise ValueError("Cannot compute eigenmodes for an empty mesh")

    want_full = k is None or int(k) >= n
    if n <= _DENSE_FULL_THRESHOLD and want_full:
        _, U, lambdas, _ = mesh_fourier_laplacian(
            vertices, faces=faces, normalized=normalized
        )
        return lambdas, U

    try:
        from scipy.sparse.linalg import eigsh
    except ImportError as e:
        raise ImportError(
            "Sparse Laplacian eigenmodes require scipy. Install with: pip install scipy"
        ) from e

    k_eff = n if k is None else min(max(int(k), 1), n)
    k_sparse = min(k_eff, max(1, n - 1))
    L = _sparse_laplacian_from_faces(faces, n, normalized=normalized)
    lambdas, U = eigsh(L, k=k_sparse, which="SM")
    order = np.argsort(lambdas)
    return lambdas[order], U[:, order]


def mesh_fourier_transform(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    normalized: bool = False,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project vertex coordinates onto Laplacian eigenmodes.

    Returns ``(coeffs, U, lambdas)``.
    """
    lambdas, U = mesh_laplacian_eigenmodes(
        vertices, faces, k=k, normalized=normalized
    )
    coeffs = U.T @ np.asarray(vertices, dtype=float)
    return coeffs, U, lambdas


def inverse_mesh_fourier(
    coeffs: np.ndarray,
    U: np.ndarray,
    k: int | None = None,
) -> np.ndarray:
    """Reconstruct vertices from graph Fourier coefficients."""
    coeffs = np.asarray(coeffs, dtype=float)
    U = np.asarray(U, dtype=float)

    if k is None:
        return U @ coeffs

    k_eff = min(int(k), coeffs.shape[0], U.shape[1])
    return U[:, :k_eff] @ coeffs[:k_eff]


def reconstruct_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    k: int,
    *,
    normalized: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Low-frequency reconstruction using the first ``k`` Laplacian modes.

    Returns ``(vertices_reconstructed, lambdas)``.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    n = int(vertices.shape[0])
    if n == 0:
        raise ValueError("Cannot reconstruct an empty mesh")

    k = min(max(int(k), 1), n)
    if n <= _DENSE_FULL_THRESHOLD and k == n:
        coeffs, U, lambdas, _ = mesh_fourier_laplacian(
            vertices, faces=faces, normalized=normalized
        )
        return inverse_mesh_fourier(coeffs, U, k=None), lambdas

    lambdas, U = mesh_laplacian_eigenmodes(
        vertices, faces, k=k, normalized=normalized
    )
    coeffs = U.T @ vertices
    k_use = min(k, U.shape[1])
    return inverse_mesh_fourier(coeffs, U, k=k_use), lambdas[:k_use]
