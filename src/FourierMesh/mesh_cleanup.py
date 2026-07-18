"""Mesh cleanup, submesh extraction, and export preparation."""

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


def _shared_edge_neighbors(faces: np.ndarray) -> list[list[int]]:
    """Face adjacency via shared undirected edges."""
    f = np.asarray(faces, dtype=np.int64)
    n_faces = int(f.shape[0])
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(n_faces):
        a, b, c = (int(x) for x in f[fi])
        for u, w in ((a, b), (b, c), (c, a)):
            e = (u, w) if u < w else (w, u)
            edge_to_faces.setdefault(e, []).append(fi)

    neigh: list[list[int]] = [[] for _ in range(n_faces)]
    for inc in edge_to_faces.values():
        if len(inc) >= 2:
            for i in inc:
                for j in inc:
                    if i != j:
                        neigh[i].append(j)
    return neigh


def submesh_max_faces(
    vertices: np.ndarray, faces: np.ndarray, max_faces: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a connected submesh with at most ``max_faces`` triangles.

    Keeps the largest shared-edge face component, then grows a BFS patch from
    its centroid so capped meshes stay connected (important for Laplacian spectra).
    """
    faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=float)
    n_faces = int(faces.shape[0])
    if n_faces == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int64)

    neigh_sets: list[set[int]] = [set() for _ in range(n_faces)]
    for i, nb_list in enumerate(_shared_edge_neighbors(faces)):
        neigh_sets[i] = set(nb_list)

    visited = np.zeros(n_faces, dtype=bool)
    largest_mask = np.zeros(n_faces, dtype=bool)
    best_size = 0
    for start in range(n_faces):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in neigh_sets[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) > best_size:
            best_size = len(comp)
            largest_mask.fill(False)
            largest_mask[np.array(comp, dtype=np.int64)] = True

    target = min(max_faces, best_size)
    comp_faces = np.flatnonzero(largest_mask)
    comp_centers = vertices[faces[comp_faces]].mean(axis=1)
    seed = int(
        comp_faces[
            np.argmin(np.linalg.norm(comp_centers - comp_centers.mean(axis=0), axis=1))
        ]
    )

    keep: list[int] = []
    queued = np.zeros(n_faces, dtype=bool)
    queued[seed] = True
    queue = [seed]
    head = 0
    while head < len(queue) and len(keep) < target:
        cur = queue[head]
        head += 1
        keep.append(cur)
        for nb in sorted(neigh_sets[cur]):
            if largest_mask[nb] and not queued[nb]:
                queued[nb] = True
                queue.append(nb)

    faces_kept = faces[np.asarray(keep, dtype=np.int64)]
    return compact_mesh(vertices, faces_kept)


def edge_multiplicity(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return undirected edge keys and occurrence counts.

    ``edges_key``: (E, 2) sorted vertex-index pairs
    ``counts``:    (E,) occurrence count of each undirected edge in faces
    """
    f = np.asarray(faces, dtype=np.int64)
    if f.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)
    edges_key, counts = np.unique(e, axis=0, return_counts=True)
    return edges_key, counts


def face_component_count(faces: np.ndarray) -> int:
    """Count shared-edge connected face components."""
    f = np.asarray(faces, dtype=np.int64)
    n_faces = int(f.shape[0])
    if n_faces == 0:
        return 0

    neigh = _shared_edge_neighbors(f)
    visited = np.zeros(n_faces, dtype=bool)
    components = 0
    for start in range(n_faces):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        while stack:
            cur = stack.pop()
            for nb in neigh[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
    return components


def keep_largest_face_component(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop disconnected face islands created by cleanup or repair."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    n_faces = int(f.shape[0])
    if n_faces == 0:
        return v, f

    neigh = _shared_edge_neighbors(f)
    visited = np.zeros(n_faces, dtype=bool)
    largest: list[int] = []
    for start in range(n_faces):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in neigh[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) > len(largest):
            largest = comp

    if len(largest) == n_faces:
        return v, f
    return compact_mesh(v, f[np.asarray(largest, dtype=np.int64)])


def drop_overfull_edges(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove faces touching any edge used by more than 2 faces.

    Targets the strict non-manifold condition slicers usually report.
    Iterates until no edge has multiplicity > 2.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if f.shape[0] == 0:
        return v, f

    max_iter = max(1, 5 * f.shape[0])
    for _ in range(max_iter):
        edges_key, counts = edge_multiplicity(f)
        bad = edges_key[counts > 2]
        if bad.shape[0] == 0:
            return v, f

        # Score each face by how many of its 3 edges are overfull. Pack each
        # sorted edge into a single int key so membership is a vectorized isin.
        stride = int(f.max()) + 1
        bad_keys = bad[:, 0] * stride + bad[:, 1]
        bad_score = np.zeros(f.shape[0], dtype=np.int64)
        for pair in (f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]):
            e = np.sort(pair, axis=1)
            bad_score += np.isin(e[:, 0] * stride + e[:, 1], bad_keys)

        if not bad_score.any() or f.shape[0] <= 1:
            return v, f
        f = np.delete(f, int(np.argmax(bad_score)), axis=0)

    return v, f


def prepare_mesh_for_export(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    use_trimesh: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sanitize mesh before STL export (slicer-friendly).

    Removes degenerate and duplicate faces, drops overfull edges, optionally
    runs ``trimesh`` repair (merge vertices, consistent normals, fill holes).
    """
    v, f = remove_degenerate_faces(vertices, faces)
    v, f = remove_duplicate_faces(v, f)
    v, f = drop_overfull_edges(v, f)
    v, f = compact_mesh(v, f)
    if f.shape[0] == 0:
        raise ValueError("Mesh is empty after cleanup; try different k or max_faces.")

    if use_trimesh:
        try:
            import trimesh
        except ImportError:
            print(
                "Note: trimesh not installed; for better slicer compatibility run: "
                "pip install trimesh  (or pip install -e \".[slicer]\" from repo root)"
            )
            use_trimesh = False
        if use_trimesh:
            tm = trimesh.Trimesh(
                vertices=v.astype(np.float64),
                faces=f.astype(np.int64),
                process=True,
            )
            trimesh.repair.fix_winding(tm)
            trimesh.repair.fix_normals(tm, multibody=True)
            trimesh.repair.fill_holes(tm)
            v = np.asarray(tm.vertices, dtype=float)
            f = np.asarray(tm.faces, dtype=np.int64)
            if f.shape[0] == 0:
                raise ValueError("Mesh is empty after trimesh repair.")

    v, f = remove_degenerate_faces(v, f)
    v, f = remove_duplicate_faces(v, f)
    v, f = drop_overfull_edges(v, f)
    v, f = keep_largest_face_component(v, f)
    v, f = compact_mesh(v, f)
    if f.shape[0] == 0:
        raise ValueError("Mesh is empty after final cleanup; try a larger k.")

    return v, f
