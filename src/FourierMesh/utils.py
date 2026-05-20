import numpy as np
from stl import mesh

def cartesian_DFT_dirac(points:list[tuple[int, int, int]], kx, ky, kz):
    """Represent points as sum of dirac deltas in 3d, so then its a continuous function, and we can approximate it with fourier transform
    """
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    kz = np.asarray(kz, dtype=float)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k_grid = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], axis=1)
    phase = k_grid @ points.T
    F_flat = np.exp(-1j * phase) @ np.ones(n_points, dtype=float)

    F = F_flat.reshape(len(kx), len(ky), len(kz))
    return F

def mesh_fourier_laplacian(vertices, faces=None, edges=None, normalized=False):
    """
    Graph/mesh Fourier transform using Laplacian eigenvectors.

    vertices: (N, 3) array of xyz vertex positions
    faces:    (M, 3) array of triangle indices, optional
    edges:    (E, 2) array of edge indices, optional
    normalized: whether to use normalized graph Laplacian

    returns:
        coeffs: Fourier coefficients of shape (N, 3)
        U:      eigenvector matrix of shape (N, N)
        lambdas:eigenvalues of shape (N,)
        L:      Laplacian matrix
    """

    vertices = np.asarray(vertices, dtype=float)
    N = vertices.shape[0]

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")

    W = np.zeros((N, N), dtype=float)

    # Build adjacency from triangle faces
    if faces is not None:
        faces = np.asarray(faces, dtype=int)

        for tri in faces:
            i, j, k = tri

            W[i, j] = 1
            W[j, i] = 1

            W[j, k] = 1
            W[k, j] = 1

            W[k, i] = 1
            W[i, k] = 1

    # Or build adjacency from edges directly
    elif edges is not None:
        edges = np.asarray(edges, dtype=int)

        for i, j in edges:
            W[i, j] = 1
            W[j, i] = 1

    else:
        raise ValueError("Pass either faces or edges")

    degrees = np.sum(W, axis=1)
    D = np.diag(degrees)

    if normalized:
        inv_sqrt_deg = np.zeros_like(degrees)
        nonzero = degrees > 0
        inv_sqrt_deg[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

        D_inv_sqrt = np.diag(inv_sqrt_deg)
        L = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = D - W

    # Eigen-decomposition of graph Laplacian
    lambdas, U = np.linalg.eigh(L)

    # Sort from low frequency to high frequency
    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    U = U[:, order]

    # Project vertex coordinates onto graph Fourier basis
    coeffs = U.T @ vertices

    return coeffs, U, lambdas, L


def inverse_mesh_fourier(coeffs, U, k=None):
    """
    Reconstruct vertices from graph Fourier coefficients.

    If k is given, keep only the first k low-frequency modes.
    """

    coeffs = np.asarray(coeffs, dtype=float)

    if k is None:
        return U @ coeffs

    coeffs_low = np.zeros_like(coeffs)
    coeffs_low[:k, :] = coeffs[:k, :]

    return U @ coeffs_low


def load_stl_as_vertices_faces(filename, dedupe=True, decimals=6):
    """
    Loads an STL file and returns:

        vertices: (N, 3)
        faces:    (M, 3)

    These can be passed into mesh_fourier_laplacian(vertices, faces).
    """

    stl_mesh = mesh.Mesh.from_file(filename)

    raw_vertices = stl_mesh.vectors.reshape(-1, 3)

    if not dedupe:
        vertices = raw_vertices
        faces = np.arange(len(raw_vertices)).reshape(-1, 3)
        return vertices, faces

    # STL repeats vertices per triangle, so we merge identical/near-identical ones.
    rounded = np.round(raw_vertices, decimals=decimals)

    vertices, inverse = np.unique(
        rounded,
        axis=0,
        return_inverse=True
    )

    faces = inverse.reshape(-1, 3)

    return vertices.astype(float), faces.astype(int)

def get_point_cloud_from_stl(file_path: str, num_points: int) -> np.ndarray:
    """Load an STL file and sample points from its surface.
    """
    stl_mesh = mesh.Mesh.from_file(file_path)
    triangles = stl_mesh.vectors

    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    cross_prod = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    
    triangle_indices = np.random.choice(len(triangles), size=num_points, p=areas/areas.sum())
    
    sampled_points = []
    for idx in triangle_indices:
        a, b, c = triangles[idx]
        r1, r2 = np.random.rand(2)
        point = (1 - np.sqrt(r1)) * a + (np.sqrt(r1) * (1 - r2)) * b + (np.sqrt(r1) * r2) * c
        sampled_points.append(point)

    return np.array(sampled_points)

def normalize_points(points:np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    points = points - center
    scale = np.max(maxs - mins) / 2
    points = points / scale
    return points