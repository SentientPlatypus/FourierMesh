"""FourierMesh add-on: spectral smoothing and eigenmode visualization."""

# Legacy metadata; Blender 4.2+ reads blender_manifest.toml instead.
bl_info = {
    "name": "FourierMesh",
    "author": "Geneustace Wicaksono",
    "version": (0, 1, 1),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > FourierMesh",
    "description": "Graph-Laplacian spectral smoothing and eigenmode visualization for meshes",
    "category": "Mesh",
}

import bpy

from . import operators, ui

_classes = (*operators.classes, *ui.classes)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    operators._BASIS_CACHE.clear()


if __name__ == "__main__":
    register()
