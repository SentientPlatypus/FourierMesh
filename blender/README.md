# FourierMesh — Blender add-on

Graph-Laplacian spectral smoothing and eigenmode visualization, inside Blender.

This is a self-contained, **numpy-only** port of the FourierMesh dense path. It
needs no scipy and no pip install — it runs on the numpy that ships with
Blender.

<video src="https://github.com/SentientPlatypus/FourierMesh/raw/master/tests/Artifacts/david_fourier_smoothing.mp4" controls muted width="100%"></video>

*Spectral smoothing of the David statue, `k` swept live in this add-on. ([Download the clip](../tests/Artifacts/david_fourier_smoothing.mp4).)*

## What it does

- **Spectral Smooth** — reconstruct the active mesh from its first `k`
  graph-Laplacian modes. Low `k` → smooth, blobby, global shape; high `k` →
  back toward the original detail. The eigenbasis is solved **once** per mesh
  and cached, so dragging `k` in the redo panel is instant.
- **Visualize Eigenmode** — paint a single eigenvector (a "frequency") onto the
  mesh as a color attribute, blue→white→red. The clearest way to *see* what a
  mesh frequency is.

## Install

**Blender 4.2+ (Extensions):**

1. Zip the add-on folder so the zip contains `fouriermesh_blender/` at its root:
   ```bash
   cd blender
   zip -r fouriermesh_blender.zip fouriermesh_blender
   ```
2. Blender → `Edit > Preferences > Add-ons` → the ▾ menu (top-right) →
   **Install from Disk…** → pick the zip.
3. Enable **FourierMesh** if it isn't already.

**Blender 4.0–4.1 (legacy):** same, via `Edit > Preferences > Add-ons >
Install…`.

## Usage

1. Select a **mesh object** and stay in **Object Mode**.
2. Open the sidebar: `View3D` → press **N** → **FourierMesh** tab.
3. Click **Spectral Smooth**. Then press **F9** (or use the bottom-left redo
   panel) to drag **Modes (k)** and watch the mesh resolve live.
4. Click **Visualize Eigenmode**, set **Mode** in the redo panel, and switch the
   viewport shading **Color** to **Attribute** to see the painted frequency.
   Undo (Ctrl+Z) restores the original positions.

## Limits (this MVP)

- **Dense solver, no vertex cap — but mind the cost.** The eigendecomposition is
  `O(N³)` in time and `O(N²)` in memory, and it runs synchronously (Blender's UI
  is frozen while it solves, with no progress bar). In practice a few thousand
  verts is instant, ~10–15k takes seconds-to-minutes, and beyond ~20k you risk
  running out of memory or a hard freeze. There's no guard rail — decimate big
  meshes, or use the sparse (scipy `eigsh`) path, which only ever solves the
  lowest `k` modes and never forms the full N×N matrix.
- Operates on **topological connectivity** from `mesh.edges` (no triangulation
  needed; quads/n-gons are fine). Disconnected pieces are smoothed
  independently — expect islands to shrink toward their own low-frequency shape.
- Destructive: it writes vertex positions in place. Ctrl+Z to revert.

## Layout

```
fouriermesh_blender/
  __init__.py             # bl_info, register/unregister
  blender_manifest.toml   # Extensions manifest (Blender 4.2+)
  spectral.py             # numpy-only Laplacian solve + reconstruct
  operators.py            # Spectral Smooth + Visualize Eigenmode
  ui.py                   # N-panel
```

`spectral.py` mirrors the dense path of the main library (`src/FourierMesh`);
the sparse solver and mesh cleanup live there.
