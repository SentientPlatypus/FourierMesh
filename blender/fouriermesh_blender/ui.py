"""Sidebar (N-panel) UI for the FourierMesh add-on."""

from __future__ import annotations

import bpy


class FOURIERMESH_PT_panel(bpy.types.Panel):
    bl_label = "FourierMesh"
    bl_idname = "FOURIERMESH_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FourierMesh"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        if obj is None or obj.type != "MESH":
            layout.label(text="Select a mesh object", icon="INFO")
            return
        if obj.mode != "OBJECT":
            layout.label(text="Switch to Object Mode", icon="INFO")
            return

        col = layout.column(align=True)
        col.operator("fouriermesh.spectral_smooth", icon="MOD_SMOOTH")
        col.operator("fouriermesh.show_eigenmode", icon="COLOR")

        box = layout.box()
        box.label(text=f"Vertices: {len(obj.data.vertices)}", icon="VERTEXSEL")
        box.label(text="Tune k / mode in the redo panel (F9)", icon="INFO")


classes = (FOURIERMESH_PT_panel,)
