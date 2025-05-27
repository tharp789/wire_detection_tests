import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def create_cylinder_between_points(p1, p2, display_length = 5.0, radius=0.005, resolution=20, color=(1, 0, 0)):
    # Compute vector between points
    vec = p2 - p1
    length = np.linalg.norm(vec)
    direction = vec / length

    # Create cylinder
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=display_length, resolution=resolution)
    cylinder.paint_uniform_color(color)

    # Compute transformation
    midpoint = (p1 + p2) / 2
    cylinder.translate(midpoint)

    # Compute rotation from Z axis to direction vector
    z_axis = np.array([0, 0, 1])
    rotation_vector = np.cross(z_axis, direction)
    dot = np.dot(z_axis, direction)

    if np.linalg.norm(rotation_vector) < 1e-6:
        # Vectors are aligned or opposite
        if dot > 0.999:
            rot = np.eye(3)
        else:
            # 180-degree flip around arbitrary axis perpendicular to Z
            rot = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
    else:
        angle = np.arccos(dot)
        rot = R.from_rotvec(angle * rotation_vector / np.linalg.norm(rotation_vector)).as_matrix()

    cylinder.rotate(rot, center=midpoint)
    return cylinder

def create_depth_viz(depth, min_depth=0.5, max_depth=20.0):
    depth[np.isnan(depth)] = min_depth
    depth[np.isinf(depth)] = min_depth
    depth[np.isneginf(depth)] = min_depth

    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = (depth * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth
