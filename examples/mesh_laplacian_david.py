"""Graph Laplacian spectrum on David STL: original vs low-frequency reconstruction.

Run from repo root::

    python examples/mesh_laplacian_david.py

Requires ``matplotlib`` (see ``[dev]`` extras in ``pyproject.toml``) and ``scipy``.

The STL is resolved in order: ``FOURIERMESH_DAVID_STL``, ``tests/models/DavidStatue.stl``,
``tests/fixtures/DavidStatue.stl`` (same layout as the David mesh test).

Eigendecomposition is ``O(N^3)``; use ``--max-faces`` to cap triangle count for interactive use.

Writes the plot to ``tests/Artifacts/Mesh/david_mesh_fourier_compare_k_<k>.png`` and the
reconstructed mesh to ``tests/models/reconstructed_david_k_<k>_.stl`` by default
(override with ``--output`` and ``--output-stl``) using ``numpy-stl``. Before export,
faces are cleaned (degenerate / duplicate removal); install ``trimesh``
(``pip install -e ".[slicer]"``) and keep the default repair path to improve winding
for slicers (e.g. Bambu Studio). Use ``--no-repair`` to skip ``trimesh``. Use
``--stl-format ascii`` if needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if _src.is_dir():
    _s = str(_src)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from FourierMesh import (  # noqa: E402
    load_mesh_stl,
    reconstruct_mesh,
    save_mesh_stl,
)
from FourierMesh.mesh_cleanup import (  # noqa: E402
    compact_mesh,
    remove_degenerate_faces,
    remove_duplicate_faces,
)


def _david_stl_path() -> Path:
    env = os.environ.get("FOURIERMESH_DAVID_STL")
    if env:
        return Path(env).expanduser().resolve()
    for rel in (
        Path("tests") / "models" / "DavidStatue.stl",
        Path("tests") / "fixtures" / "DavidStatue.stl",
    ):
        p = (_repo_root / rel).resolve()
        if p.is_file():
            return p
    return _repo_root / "tests" / "fixtures" / "DavidStatue.stl"


def _submesh(
    vertices: np.ndarray, faces: np.ndarray, max_faces: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample faces and remap vertices to 0..n-1.

    Naïvely taking ``faces[:max_faces]`` can include triangles from several disconnected
    island components. A BFS “first N faces” from one seed only covers a **small local
    patch** on a large mesh.

    Here we: (1) build shared-edge adjacency, (2) keep the **largest connected component**
    of faces, (3) take up to ``max_faces`` triangles by BFS within that component. The
    cap is now an explicit debug/speed mode, so keeping the capped mesh connected is more
    important than trying to sample the whole statue.
    """
    faces = np.asarray(faces, dtype=np.int64)
    n_faces = int(faces.shape[0])
    if n_faces == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int64)

    # Build edge -> incident faces
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(n_faces):
        a, b, c = (int(x) for x in faces[fi])
        for u, w in ((a, b), (b, c), (c, a)):
            e = (u, w) if u < w else (w, u)
            edge_to_faces.setdefault(e, []).append(fi)

    # Shared-edge adjacency (two faces on the same undirected edge)
    neigh: list[set[int]] = [set() for _ in range(n_faces)]
    for inc in edge_to_faces.values():
        if len(inc) >= 2:
            for i in inc:
                for j in inc:
                    if i != j:
                        neigh[i].add(j)

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
            for nb in neigh[cur]:
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
    seed = int(comp_faces[np.argmin(np.linalg.norm(comp_centers - comp_centers.mean(axis=0), axis=1))])

    keep: list[int] = []
    queued = np.zeros(n_faces, dtype=bool)
    queued[seed] = True
    queue = [seed]
    head = 0
    while head < len(queue) and len(keep) < target:
        cur = queue[head]
        head += 1
        keep.append(cur)
        for nb in sorted(neigh[cur]):
            if largest_mask[nb] and not queued[nb]:
                queued[nb] = True
                queue.append(nb)

    faces_kept = faces[np.asarray(keep, dtype=np.int64)]
    used = np.unique(faces_kept.ravel())
    v_sub = np.asarray(vertices, dtype=float)[used]
    old_to_new = -np.ones(vertices.shape[0], dtype=np.int64)
    old_to_new[used] = np.arange(used.size, dtype=np.int64)
    f_sub = old_to_new[faces_kept]
    return v_sub, f_sub


def _edge_multiplicity(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return undirected edge keys and counts.

    edges_key: (E, 2) sorted vertex-index pairs
    counts:    (E,) occurrence count of each undirected edge in faces
    """
    f = np.asarray(f, dtype=np.int64)
    if f.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)
    edges_key, counts = np.unique(e, axis=0, return_counts=True)
    return edges_key, counts


def _face_component_count(f: np.ndarray) -> int:
    """Count shared-edge connected face components."""
    f = np.asarray(f, dtype=np.int64)
    n_faces = int(f.shape[0])
    if n_faces == 0:
        return 0

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


def _keep_largest_face_component(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop disconnected face islands created by cleanup/repair."""
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    n_faces = int(f.shape[0])
    if n_faces == 0:
        return v, f

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


def _drop_overfull_edges(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove faces touching any edge used by more than 2 faces.

    This targets the strict non-manifold condition slicers usually report.
    Iterates until no edge has multiplicity > 2.
    """
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=np.int64)
    if f.shape[0] == 0:
        return v, f

    max_iter = max(1, 5 * f.shape[0])
    for _ in range(max_iter):
        edges_key, counts = _edge_multiplicity(f)
        bad = edges_key[counts > 2]
        if bad.shape[0] == 0:
            return v, f

        bad_set = {tuple(e) for e in bad.tolist()}
        e01 = np.sort(f[:, [0, 1]], axis=1)
        e12 = np.sort(f[:, [1, 2]], axis=1)
        e20 = np.sort(f[:, [2, 0]], axis=1)
        bad_score = np.array(
            [
                int(tuple(e01[i].tolist()) in bad_set)
                + int(tuple(e12[i].tolist()) in bad_set)
                + int(tuple(e20[i].tolist()) in bad_set)
                for i in range(f.shape[0])
            ],
            dtype=np.int64,
        )
        candidate = np.flatnonzero(bad_score > 0)
        if candidate.size == 0:
            return v, f
        # Remove one face that participates in the most overfull edges.
        drop_idx = int(candidate[np.argmax(bad_score[candidate])])
        if f.shape[0] <= 1:
            return v, f
        f = np.delete(f, drop_idx, axis=0)

    return v, f


def _prepare_mesh_for_export(
    v: np.ndarray,
    f: np.ndarray,
    *,
    use_trimesh: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sanitize mesh before STL export (slicer-friendly): degenerate / duplicate removal,
    optional ``trimesh`` process (merge, consistent normals).
    """
    v, f = remove_degenerate_faces(v, f)
    v, f = remove_duplicate_faces(v, f)
    v, f = _drop_overfull_edges(v, f)
    v, f = compact_mesh(v, f)
    if f.shape[0] == 0:
        raise ValueError("Mesh is empty after cleanup; try different --k or --max-faces.")

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
    v, f = _drop_overfull_edges(v, f)
    v, f = _keep_largest_face_component(v, f)
    v, f = compact_mesh(v, f)
    if f.shape[0] == 0:
        raise ValueError("Mesh is empty after final cleanup; try a larger --k.")

    return v, f


# Match ``tests/Artifacts/Mesh/`` comparison plot styling.
_FIG_FACE = "#1a1a2e"
_AX_FACE = "#1a1a2e"
_CMAP_NAME = "inferno"


def _plot_pair(
    v_orig: np.ndarray,
    v_recon: np.ndarray,
    faces: np.ndarray,
    k: int,
    out_path: Path | None,
    mesh_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for this example. Install with: pip install matplotlib"
        ) from e

    faces = np.asarray(faces, dtype=int)
    cmap = colormaps[_CMAP_NAME]

    def _triangle_polys(verts: np.ndarray) -> np.ndarray:
        """(F, 3, 3): each row is one triangle's three corner positions."""
        v = np.asarray(verts, dtype=float)
        return v[faces]

    def _face_colors(verts: np.ndarray) -> np.ndarray:
        tri = _triangle_polys(verts)
        zmean = tri[:, :, 2].mean(axis=1)
        z0, z1 = float(zmean.min()), float(zmean.max())
        if z1 <= z0 + 1e-15:
            t = np.full(len(faces), 0.5)
        else:
            t = (zmean - z0) / (z1 - z0)
        return cmap(t)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        subplot_kw={"projection": "3d"},
        facecolor=_FIG_FACE,
    )
    fig.patch.set_facecolor(_FIG_FACE)

    both = np.vstack([v_orig, v_recon])
    lo, hi = both.min(axis=0), both.max(axis=0)
    c = 0.5 * (lo + hi)
    ptp = hi - lo
    ptp = np.where(ptp < 1e-12 * (float(np.max(ptp)) + 1.0), 1.0, ptp)
    # Equal data scale on x and y (same half-width on both axes, as in cube_dft 3D panels).
    half_xy = 0.5 * max(ptp[0], ptp[1]) * 1.05
    half_z = 0.5 * ptp[2] * 1.05
    box = (half_xy, half_xy, max(half_z, 1e-12 * half_xy))

    def _style_dark_3d(ax) -> None:
        ax.set_facecolor(_AX_FACE)
        ax.tick_params(colors="0.92", labelsize=9)
        ax.xaxis.label.set_color("0.92")
        ax.yaxis.label.set_color("0.92")
        ax.zaxis.label.set_color("0.92")
        ax.title.set_color("0.95")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("0.55")
            axis.pane.set_alpha(0.35)
        ax.grid(True, color="0.85", linestyle="-", linewidth=0.35, alpha=0.22)

    def _one(ax, verts: np.ndarray, title: str) -> None:
        _style_dark_3d(ax)
        polys = _triangle_polys(verts)
        coll = Poly3DCollection(
            polys,
            facecolors=_face_colors(verts),
            edgecolor=(1.0, 1.0, 1.0, 0.12),
            linewidths=0.1,
            alpha=0.95,
        )
        ax.add_collection3d(coll)
        zc = verts[:, 2]
        z0, z1 = float(zc.min()), float(zc.max())
        if z1 <= z0 + 1e-15:
            sc = np.full(len(verts), 0.5)
        else:
            sc = (zc - z0) / (z1 - z0)
        ax.scatter(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            c=cmap(sc),
            s=5,
            alpha=0.45,
            edgecolors="none",
            depthshade=False,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(c[0] - half_xy, c[0] + half_xy)
        ax.set_ylim(c[1] - half_xy, c[1] + half_xy)
        ax.set_zlim(c[2] - half_z, c[2] + half_z)
        ax.set_box_aspect(box)
        ax.view_init(elev=18, azim=-58)

    _one(axes[0], v_orig, f"Original ({mesh_label})")
    _one(axes[1], v_recon, f"Reconstructed (k={k} lowest graph-Fourier modes)")
    fig.suptitle(
        "reconstruct_mesh (graph Laplacian low-pass)",
        fontsize=11,
        color="0.95",
    )
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=150,
            facecolor=_FIG_FACE,
            edgecolor="none",
        )
        print(f"Wrote {out_path}")
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="David STL: Laplacian spectrum reconstruction")
    p.add_argument("--k", type=int, default=24, help="Number of lowest-frequency modes to keep")
    p.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help="Optional cap on triangle count for faster runs; omit to reconstruct the full mesh",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="PNG path for the comparison plot (default: tests/Artifacts/Mesh/david_mesh_fourier_compare_k_<k>.png)",
    )
    p.add_argument(
        "--output-stl",
        type=Path,
        default=None,
        help="Reconstructed mesh STL path (default: tests/models/reconstructed_david_k_<k>_.stl)",
    )
    p.add_argument(
        "--no-save-stl",
        action="store_true",
        help="Do not write reconstructed mesh to STL",
    )
    p.add_argument(
        "--stl-format",
        choices=("binary", "ascii"),
        default="binary",
        help="STL encoding (default binary; try ascii for strict viewers)",
    )
    p.add_argument(
        "--no-repair",
        action="store_true",
        help="Skip trimesh-based repair (still removes degenerate/duplicate faces in numpy)",
    )
    args = p.parse_args()

    stl_path = _david_stl_path()
    if not stl_path.is_file():
        raise SystemExit(
            f"STL not found: {stl_path}\n"
            "Place DavidStatue.stl under tests/models/ or tests/fixtures/, "
            "or set FOURIERMESH_DAVID_STL."
        )

    vertices, faces = load_mesh_stl(str(stl_path), dedupe=True)
    if args.max_faces is not None:
        if args.max_faces <= 0:
            raise SystemExit("--max-faces must be a positive integer when provided")
        v_mesh, f_mesh = _submesh(vertices, faces, max_faces=args.max_faces)
        mesh_label = f"submesh, {f_mesh.shape[0]} of {faces.shape[0]} faces"
    else:
        v_mesh, f_mesh = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
        mesh_label = "full mesh"
    n = v_mesh.shape[0]

    k_eff = min(max(args.k, 1), n)
    if k_eff != args.k:
        print(f"Note: clamped k from {args.k} to {k_eff} (vertex count N={n})")

    v_recon, lambdas = reconstruct_mesh(v_mesh, f_mesh, k=k_eff, normalized=False)
    err = np.linalg.norm(v_mesh - v_recon) / (np.linalg.norm(v_mesh) + 1e-12)
    print(f"Mesh scope: {mesh_label}")
    print(f"Vertices N={n}, faces={f_mesh.shape[0]}, k={k_eff}, relative L2 error vs full: {err:.4g}")

    if not args.no_save_stl:
        stl_out = args.output_stl
        if stl_out is None:
            stl_out = _repo_root / "tests" / "models" / f"reconstructed_david_k_{k_eff}_.stl"
        else:
            stl_out = Path(stl_out).expanduser().resolve()
        v_exp, f_exp = _prepare_mesh_for_export(
            v_recon,
            f_mesh,
            use_trimesh=not args.no_repair,
        )
        _, edge_counts = _edge_multiplicity(f_exp)
        nonmanifold_edge_count = int(np.sum(edge_counts > 2))
        boundary_edge_count = int(np.sum(edge_counts == 1))
        component_count = _face_component_count(f_exp)
        if f_exp.shape[0] != f_mesh.shape[0]:
            print(
                f"STL export: {f_exp.shape[0]} facets after cleanup "
                f"(was {f_mesh.shape[0]} before export)"
            )
        print(
            "STL export edge stats: "
            f"nonmanifold(>2 faces)={nonmanifold_edge_count}, "
            f"boundary(1 face)={boundary_edge_count}, "
            f"components={component_count}"
        )
        save_mesh_stl(v_exp, f_exp, stl_out, fmt=args.stl_format, name="reconstructed_david")
        print(f"Wrote reconstructed STL: {stl_out}")

    plot_out = args.output
    if plot_out is None:
        plot_out = _repo_root / "tests" / "Artifacts" / "Mesh" / f"david_mesh_fourier_compare_k_{k_eff}.png"
    else:
        plot_out = Path(plot_out).expanduser().resolve()

    _plot_pair(v_mesh, v_recon, f_mesh, k_eff, plot_out, mesh_label)


if __name__ == "__main__":
    main()
