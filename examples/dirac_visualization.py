"""Run from repo root: ``python examples/dirac_visualization.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Repo checkout without ``pip install -e .``: add ``src`` so ``FourierMesh`` resolves.
_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if _src.is_dir():
    src_str = str(_src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from FourierMesh import cartesian_DFT_dirac


def hollow_cube_points(n_per_edge: int = 20, half: float = 1.0) -> np.ndarray:
    """Sample points on cube faces in [-half, half]^3."""
    lin = np.linspace(-half, half, n_per_edge)
    u, v = np.meshgrid(lin, lin)
    u, v = u.ravel(), v.ravel()
    fixed = np.full_like(u, half)
    faces = [
        np.c_[fixed, u, v],
        np.c_[-fixed, u, v],
        np.c_[u, fixed, v],
        np.c_[u, -fixed, v],
        np.c_[u, v, fixed],
        np.c_[u, v, -fixed],
    ]
    return np.unique(np.vstack(faces), axis=0)


def plot_cube_spectrum_slice() -> None:
    points = hollow_cube_points(n_per_edge=20)
    k = np.linspace(-6.0, 6.0, 40)
    spectrum = cartesian_DFT_dirac(points, k, k, k)
    spectrum_mag = np.abs(spectrum)
    mid = len(k) // 2

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        spectrum_mag[:, :, mid].T,
        origin="lower",
        extent=[k[0], k[-1], k[0], k[-1]],
        cmap="inferno",
        interpolation="bilinear",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="|F(kx, ky, kz=0)|")
    ax.set_title("Cube point cloud Fourier magnitude slice")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_cube_spectrum_slice()
