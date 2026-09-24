"""Shared Blender helpers used by L-system and Mangrove generators."""

import bpy
import mathutils


def clear_scene():
    """Clear all objects from the current Blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def create_cylinder(start, end, radius):
    """
    Create a cylinder mesh between two 3D points to represent a tree branch.

    Returns:
        bpy.types.Object or None if the segment is too short.
    """
    v = end - start
    length = v.length
    if length < 1e-4:
        return None

    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length)
    cyl = bpy.context.active_object

    z_axis = mathutils.Vector((0, 0, 1))
    v_norm = v.normalized()
    quat = z_axis.rotation_difference(v_norm)

    cyl.rotation_mode = "QUATERNION"
    cyl.rotation_quaternion = quat
    cyl.location = start + v / 2.0

    return cyl
