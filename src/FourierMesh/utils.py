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