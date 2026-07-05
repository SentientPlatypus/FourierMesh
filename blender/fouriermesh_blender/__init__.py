"""FourierMesh -- graph-Laplacian spectral smoothing for Blender meshes.

Self-contained add-on: runs on Blender's bundled numpy, no external deps.
See ``blender/README.md`` for install and usage.
"""

# bl_info supports the legacy add-on installer (Blender 4.0-4.1 and drop-in
# installs). Blender 4.2+ installs via the Extensions system and reads
# blender_manifest.toml instead; bl_info is ignored there.
bl_info = {
    "name": "FourierMesh",
    "author": "Geneustace Wicaksono",
    "version": (0, 1, 0),
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
