"""Blender operators for FourierMesh spectral smoothing and eigenmode visualization."""

from __future__ import annotations

import bpy
import numpy as np
from bpy.props import BoolProperty, IntProperty

from . import spectral

# mesh-datablock name -> {"n", "n_edges", "normalized", "U", "coeffs", "lambdas"}
# The basis is solved once (in invoke / on cache miss) and reused so dragging k
# in the redo panel is instant.
_BASIS_CACHE: dict[str, dict] = {}


def _cache_key(obj: "bpy.types.Object") -> str:
    return obj.data.name


def _read_mesh_arrays(mesh: "bpy.types.Mesh") -> tuple[np.ndarray, np.ndarray]:
    """Vertex coords ``(N, 3)`` and edge endpoints ``(E, 2)`` as numpy arrays."""
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    verts = co.reshape(n, 3).astype(np.float64)

    e = len(mesh.edges)
    ev = np.empty(e * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", ev)
    edges = ev.reshape(e, 2).astype(np.int64)
    return verts, edges


def _write_mesh_positions(mesh: "bpy.types.Mesh", verts: np.ndarray) -> None:
    flat = np.asarray(verts, dtype=np.float32).reshape(-1)
    mesh.vertices.foreach_set("co", flat)
    mesh.update()


def _solve_and_cache(mesh: "bpy.types.Mesh", normalized: bool) -> dict:
    """Solve the eigenbasis for a mesh and store it (float32) in the cache."""
    verts, edges = _read_mesh_arrays(mesh)
    U, coeffs, lambdas = spectral.solve_eigenbasis(
        verts, edges, normalized=normalized
    )
    cache = {
        "n": len(mesh.vertices),
        "n_edges": len(mesh.edges),
        "normalized": normalized,
        "U": U.astype(np.float32),
        "coeffs": coeffs.astype(np.float32),
        "lambdas": lambdas,
    }
    _BASIS_CACHE[mesh.name] = cache
    return cache


def _cache_is_valid(cache: dict | None, mesh: "bpy.types.Mesh", normalized: bool) -> bool:
    return (
        cache is not None
        and cache["n"] == len(mesh.vertices)
        and cache["n_edges"] == len(mesh.edges)
        and cache["normalized"] == normalized
    )


def _mesh_poll(context) -> bool:
    obj = context.active_object
    return obj is not None and obj.type == "MESH" and obj.mode == "OBJECT"


class FOURIERMESH_OT_spectral_smooth(bpy.types.Operator):
    """Low-pass reconstruct the mesh using the first k graph-Laplacian modes"""

    bl_idname = "fouriermesh.spectral_smooth"
    bl_label = "Spectral Smooth"
    bl_options = {"REGISTER", "UNDO"}

    k: IntProperty(
        name="Modes (k)",
        description="Number of low-frequency Laplacian modes to keep. "
        "Fewer = smoother/blobbier, more = closer to the original. "
        "Clamped to the vertex count at run time",
        default=24,
        min=1,
    )
    normalized: BoolProperty(
        name="Normalized Laplacian",
        description="Use the symmetric normalized Laplacian instead of the "
        "combinatorial one",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return _mesh_poll(context)

    def invoke(self, context, event):
        # A fresh click just needs a valid basis; execute() solves it on a cache
        # miss and reuses it otherwise. Reusing keeps re-clicks instant AND
        # idempotent: reconstruction always starts from the ORIGINAL geometry
        # captured on the first solve, so clicking again never over-smooths.
        # (Topology changes or flipping "normalized" invalidate the cache and
        # force a fresh solve; see _cache_is_valid.)
        mesh = context.active_object.data
        n = len(mesh.vertices)
        if n == 0:
            self.report({"ERROR"}, "Mesh has no vertices")
            return {"CANCELLED"}
        self.k = min(self.k, n)
        return self.execute(context)

    def execute(self, context):
        mesh = context.active_object.data
        cache = _BASIS_CACHE.get(mesh.name)
        if not _cache_is_valid(cache, mesh, self.normalized):
            cache = _solve_and_cache(mesh, self.normalized)

        recon = spectral.reconstruct(cache["U"], cache["coeffs"], self.k)
        _write_mesh_positions(mesh, recon)
        self.report(
            {"INFO"},
            f"Spectral smooth: k={min(self.k, cache['n'])} / {cache['n']} modes",
        )
        return {"FINISHED"}


def _diverging_colors(values: np.ndarray) -> np.ndarray:
    """Map a signed per-vertex scalar to a blue-white-red RGBA array ``(N, 4)``."""
    v = np.asarray(values, dtype=np.float64)
    amax = np.max(np.abs(v)) if v.size else 0.0
    if amax == 0.0:
        amax = 1.0
    t = v / amax  # [-1, 1]
    pos = np.clip(t, 0.0, 1.0)[:, None]
    neg = np.clip(-t, 0.0, 1.0)[:, None]
    white = np.array([1.0, 1.0, 1.0])
    red = np.array([0.84, 0.19, 0.15])
    blue = np.array([0.13, 0.35, 0.86])
    rgb = white * (1.0 - pos - neg) + red * pos + blue * neg
    rgba = np.ones((v.shape[0], 4), dtype=np.float32)
    rgba[:, :3] = rgb.astype(np.float32)
    return rgba


def _write_vertex_colors(mesh: "bpy.types.Mesh", name: str, rgba: np.ndarray) -> None:
    attrs = mesh.color_attributes
    ca = attrs.get(name)
    if ca is not None and (ca.domain != "POINT" or ca.data_type != "FLOAT_COLOR"):
        attrs.remove(ca)
        ca = None
    if ca is None:
        ca = attrs.new(name=name, type="FLOAT_COLOR", domain="POINT")
    ca.data.foreach_set("color", np.asarray(rgba, dtype=np.float32).reshape(-1))
    attrs.active_color = ca
    mesh.update()


class FOURIERMESH_OT_show_eigenmode(bpy.types.Operator):
    """Paint a single graph-Laplacian eigenvector onto the mesh as a color attribute"""

    bl_idname = "fouriermesh.show_eigenmode"
    bl_label = "Visualize Eigenmode"
    bl_options = {"REGISTER", "UNDO"}

    mode_index: IntProperty(
        name="Mode",
        description="Which eigenvector (frequency) to paint. 0 is the flat DC "
        "mode; higher indices are higher-frequency detail",
        default=1,
        min=0,
        soft_max=64,
    )
    normalized: BoolProperty(
        name="Normalized Laplacian",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return _mesh_poll(context)

    def execute(self, context):
        mesh = context.active_object.data
        if len(mesh.vertices) == 0:
            self.report({"ERROR"}, "Mesh has no vertices")
            return {"CANCELLED"}

        cache = _BASIS_CACHE.get(mesh.name)
        if not _cache_is_valid(cache, mesh, self.normalized):
            cache = _solve_and_cache(mesh, self.normalized)

        m = max(0, min(self.mode_index, cache["n"] - 1))
        colors = _diverging_colors(cache["U"][:, m])
        _write_vertex_colors(mesh, "FourierMode", colors)
        self.report(
            {"INFO"},
            f"Eigenmode {m} (lambda={cache['lambdas'][m]:.4f}) - set viewport "
            "Color to 'Attribute' to view",
        )
        return {"FINISHED"}


classes = (
    FOURIERMESH_OT_spectral_smooth,
    FOURIERMESH_OT_show_eigenmode,
)
