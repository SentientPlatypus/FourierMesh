"""Public package API for FourierMesh — spectral mesh analysis via graph Laplacian."""

from .io import load_mesh_stl, load_stl_as_vertices_faces, save_mesh_stl
from .laplacian import (
    inverse_mesh_fourier,
    mesh_fourier_laplacian,
    mesh_fourier_transform,
    mesh_laplacian_eigenmodes,
    reconstruct_mesh,
)
from .mesh_cleanup import (
    compact_mesh,
    drop_overfull_edges,
    edge_multiplicity,
    face_component_count,
    keep_largest_face_component,
    prepare_mesh_for_export,
    remove_degenerate_faces,
    remove_duplicate_faces,
    submesh_max_faces,
)

__all__ = [
    "load_mesh_stl",
    "load_stl_as_vertices_faces",
    "save_mesh_stl",
    "mesh_fourier_laplacian",
    "mesh_fourier_transform",
    "mesh_laplacian_eigenmodes",
    "inverse_mesh_fourier",
    "reconstruct_mesh",
    "compact_mesh",
    "drop_overfull_edges",
    "edge_multiplicity",
    "face_component_count",
    "keep_largest_face_component",
    "prepare_mesh_for_export",
    "remove_degenerate_faces",
    "remove_duplicate_faces",
    "submesh_max_faces",
]
