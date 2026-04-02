import sys, os #REMOVE THIS BEFORE PUBLISHING
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


import numpy as np
import matplotlib.pyplot as plt
from FourierMesh.utils import cartesian_DFT_dirac, get_point_cloud_from_stl, normalize_points

def hollow_cube_points(n_per_edge=20, half=1.0):
    """Sample points on the 6 faces of a cube [-half, half]^3."""
    lin = np.linspace(-half, half, n_per_edge)
    u, v = np.meshgrid(lin, lin)
    u, v = u.ravel(), v.ravel()
    fixed = np.full_like(u, half)

    faces = [
        np.c_[fixed,  u, v],   # x = +1
        np.c_[-fixed, u, v],   # x = -1
        np.c_[u, fixed,  v],   # y = +1
        np.c_[u, -fixed, v],   # y = -1
        np.c_[u, v,  fixed],   # z = +1
        np.c_[u, v, -fixed],   # z = -1
    ]
    return np.unique(np.vstack(faces), axis=0)   # deduplicate shared edges

def hollow_sphere_points(n_points=2000, radius=1.0):
    """Sample points approximately uniformly on a sphere surface
    using the Fibonacci lattice (golden angle spiral).
    
    Parameters
    ----------
    n_points : int
        Number of points on the sphere surface.
    radius : float
        Sphere radius.
    """
    golden = np.pi * (3 - np.sqrt(5))          # golden angle ~2.399 rad

    i = np.arange(n_points)
    y = 1 - (i / (n_points - 1)) * 2           # y from +1 to -1
    r = np.sqrt(1 - y**2)                       # radius at each y slice

    theta = golden * i                          # azimuthal angle
    x = np.cos(theta) * r
    z = np.sin(theta) * r

    points = np.stack([x, y, z], axis=1) * radius
    return points

def upper_hemisphere_points(n_points=2000, radius=1.0):
    points = []
    while len(points) < n_points:
        # sample uniformly in the cube [-r, r]^3
        batch = np.random.uniform(-radius, radius, size=(n_points * 2, 3))
        # keep points inside the sphere and above z=0
        inside = np.sum(batch**2, axis=1) <= radius**2
        upper  = batch[:, 2] >= 0
        points.append(batch[inside & upper])

    return np.vstack(points)[:n_points]


def david_points(n_points=5000, scale=1.0,):
    filepath = "tests/models/DavidStatue.stl"
    points = get_point_cloud_from_stl(filepath, n_points)
    points = normalize_points(points)
    print(points[-10:])  # sanity check: print last few points
    test_dirac_point_cloud(points * scale, shape="david", k=np.linspace(-20, 20, 50), threshold=0.275, alpha=0.8)
     
def test_dirac_point_cloud(points, shape: str, k=np.linspace(-6, 6, 40), threshold=0.2, alpha=0.6):
    """FIRST NORMALIZE TO 1.5 cube. ChatGPT wrote this test."""
    print(f"Point cloud: {len(points)} points")

    F = cartesian_DFT_dirac(points, k, k, k)
    F_mag = np.abs(F)
    mid = len(k) // 2

    # ── figure 1: point cloud + DFT slice + base reconstruction ─────────────
    fig = plt.figure(figsize=(20, 6))
    fig.patch.set_facecolor("#1a1a2e")

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.set_facecolor("#1a1a2e")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                s=2, c=points[:, 2], cmap="plasma", alpha=0.6)
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    z_range = points[:, 2].max() - points[:, 2].min()

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_title(f"Point cloud  ({shape})", color="white", pad=10)
    ax1.set_xlabel("x", color="white"); ax1.set_ylabel("y", color="white"); ax1.set_zlabel("z", color="white")
    ax1.tick_params(colors="white")
    for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#333355")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor("#1a1a2e")
    im = ax2.imshow(F_mag[:, :, mid].T, origin="lower",
                    extent=[k[0], k[-1], k[0], k[-1]],
                    cmap="inferno", interpolation="bilinear", aspect="equal")
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    cb.set_label("|F(kx, ky, kz=0)|", color="white")
    ax2.set_title("DFT magnitude  (kz = 0 slice)", color="white", pad=10)
    ax2.set_xlabel("kx  (rad / unit)", color="white")
    ax2.set_ylabel("ky  (rad / unit)", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values(): spine.set_edgecolor("#333355")

# --- right: 3D reconstruction, color = amplitude ---
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.set_facecolor("#1a1a2e")

    n_vol = 35
    xv = np.linspace(-1.5, 1.5, n_vol)
    yv = np.linspace(-1.5, 1.5, n_vol)
    zv = np.linspace(-1.5, 1.5, n_vol)
    XV, YV, ZV = np.meshgrid(xv, yv, zv, indexing="ij")

    KX3, KY3, KZ3 = np.meshgrid(k, k, k, indexing="ij")
    w_flat  = F.ravel()
    k_all   = np.stack([KX3.ravel(), KY3.ravel(), KZ3.ravel()], axis=1)
    xyz_flat = np.stack([XV.ravel(), YV.ravel(), ZV.ravel()], axis=1)

    chunk = 5000
    vol_flat = np.zeros(len(xyz_flat))
    for i in range(0, len(xyz_flat), chunk):
        phase = xyz_flat[i:i+chunk] @ k_all.T
        vol_flat[i:i+chunk] = (np.exp(1j * phase) @ w_flat).real

    vol = vol_flat.reshape(XV.shape)
    vol = (vol - vol.min()) / (vol.max() - vol.min())

    mask = vol > threshold
    sc = ax3.scatter(XV[mask], YV[mask], ZV[mask],
                     c=vol[mask], cmap="plasma",
                     s=10, alpha=alpha, linewidths=0)

    cb3 = fig.colorbar(sc, ax=ax3, fraction=0.03, pad=0.1, shrink=0.6)
    cb3.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb3.ax.yaxis.get_ticklabels(), color="white")
    cb3.set_label("amplitude (normalised)", color="white")
    ax3.set_box_aspect([x_range, y_range, z_range])     
    ax3.set_title("Reconstruction  (3D, color = amplitude)", color="white", pad=12)
    ax3.set_xlabel("x", color="white"); ax3.set_ylabel("y", color="white"); ax3.set_zlabel("z", color="white")
    ax3.tick_params(colors="white")
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#333355")

    plt.tight_layout(pad=2.0)
    plt.savefig(f"tests/Artifacts/dirac/{shape}_dft.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()

    # ── figure 2: grid of z-slices ───────────────────────────────────────────
    z_levels = np.linspace(-1.5, 1.5, 9)
    fig2, axes = plt.subplots(3, 3, figsize=(18, 14),
                              subplot_kw={"projection": "3d"})
    fig2.patch.set_facecolor("#1a1a2e")
    fig2.suptitle(f"Sinusoid reconstruction — z slices  ({shape})", color="white", fontsize=14)

    x = np.linspace(-2, 2, 60); y = np.linspace(-2, 2, 60)
    X, Y = np.meshgrid(x, y, indexing="ij")
    KX3, KY3, KZ3 = np.meshgrid(k, k, k, indexing="ij")
    w_flat  = F.ravel()
    k_xy    = np.stack([KX3.ravel(), KY3.ravel()], axis=1)
    kz_flat = KZ3.ravel()
    xy_flat = np.stack([X.ravel(), Y.ravel()], axis=1)
    base_phase = xy_flat @ k_xy.T          # (nx*ny, nk^3) — reuse for all slices

    all_slices = []
    for z_eval in z_levels:
        phase = base_phase + kz_flat * z_eval
        recon = (np.exp(1j * phase) @ w_flat).real.reshape(X.shape)
        all_slices.append(recon)

    vmax = max(s.max() for s in all_slices)
    vmin = min(s.min() for s in all_slices)

    for ax, recon, z_eval in zip(axes.ravel(), all_slices, z_levels):
        ax.set_facecolor("#1a1a2e")
        norm = (recon - vmin) / (vmax - vmin)   # normalise for color
        ax.plot_surface(X, Y, recon,
                        facecolors=plt.get_cmap("plasma")(norm),
                        linewidth=0, antialiased=True, alpha=0.9)
        ax.set_title(f"z = {z_eval:.2f}", color="white", fontsize=10)
        ax.set_xlabel("x", color="white", fontsize=7)
        ax.set_ylabel("y", color="white", fontsize=7)
        ax.set_zlabel("amp", color="white", fontsize=7)
        ax.tick_params(colors="white", labelsize=6)
        ax.set_zlim(vmin, vmax)               # shared z scale across all slices
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False; pane.set_edgecolor("#333355")

    plt.tight_layout()
    plt.savefig(f"tests/Artifacts/dirac/{shape}_slices.png", dpi=150, bbox_inches="tight",
                facecolor=fig2.get_facecolor())
    plt.show()


def cube_point_cloud():
    points = hollow_cube_points(n_per_edge=20)
    test_dirac_point_cloud(points, shape="cube")

def hemisphere_point_cloud():
    points = upper_hemisphere_points(n_points=2000)
    test_dirac_point_cloud(points, shape="hemisphere")

def sphere_point_cloud():
    points = hollow_sphere_points(n_points=1000)
    test_dirac_point_cloud(points, shape="sphere")

if __name__ == "__main__":
    david_points()