"""Write a 1×1×1 axis-aligned cube as STL (coordinates 0..1 = 1 cm per edge).

Run from repo root::

    python examples/cube_1cm_stl.py

Default output: ``tests/models/cube_1cm.stl`` (via ``numpy-stl``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stl import Mesh, Mode

_repo_root = Path(__file__).resolve().parent.parent


def _unit_cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Vertices on [0,1]^3 and 12 triangle indices (outward winding)."""
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
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [3, 2, 6],
            [3, 6, 7],
            [0, 3, 7],
            [0, 7, 4],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return v, f


def main() -> None:
    p = argparse.ArgumentParser(description="Write 1×1×1 cube STL (0..1 units)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_repo_root / "tests" / "models" / "cube_1cm.stl",
        help="Output .stl path",
    )
    args = p.parse_args()
    out = Path(args.output).expanduser().resolve()

    v, f = _unit_cube_mesh()
    n_tri = int(f.shape[0])
    stl_data = np.zeros(n_tri, dtype=Mesh.dtype)
    stl_data["vectors"] = v[f]
    m = Mesh(
        stl_data,
        calculate_normals=True,
        remove_empty_areas=False,
        name="cube_1cm",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out), mode=Mode.BINARY, update_normals=True)
    print(f"Wrote {out} ({out.stat().st_size} bytes, {n_tri} facets, edge length 1.0)")


if __name__ == "__main__":
    main()
