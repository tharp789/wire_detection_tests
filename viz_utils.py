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

def visualize_fitted_lines(fitted_lines, roi_pcs, roi_point_colors):
    """
    Visualizes fitted lines and corresponding point clouds.
    
    Parameters:
    - fitted_lines: List of tuples containing start and end points of fitted lines.
    - roi_pcs: List of point clouds corresponding to each fitted line.
    - roi_point_colors: List of colors for each point in the point clouds.
    """
    geometries = []
    for line in fitted_lines:
        cylinder = create_cylinder_between_points(line[0], line[1], radius=0.025)
        geometries.append(cylinder)

    for points, colors in zip(roi_pcs, roi_point_colors):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(point_cloud)

    o3d.visualization.draw_geometries(geometries, 
                                      point_show_normal=True,
                                      mesh_show_back_face=True)
    
def capture_fitted_lines_in_image(fitted_lines, roi_pcs, roi_point_colors):
    """
    Captures fitted lines and corresponding point clouds in an image.
    
    Parameters:
    - fitted_lines: List of tuples containing start and end points of fitted lines.
    - roi_pcs: List of point clouds corresponding to each fitted line.
    - roi_point_colors: List of colors for each point in the point clouds.
    
    Returns:
    - image: An image with the fitted lines and point clouds visualized.
    """
    geometries = []
    for line in fitted_lines:
        cylinder = create_cylinder_between_points(line[0], line[1], radius=0.025)
        geometries.append(cylinder)

    for points, colors in zip(roi_pcs, roi_point_colors):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(point_cloud)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(geometries)
    vis.poll_events()
    vis.update_renderer()
    
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()
    
    return np.asarray(image)

def visualize_colored_point_cloud(depth_image, rgb_image, camera_intrinsics):
    min_depth = 0.5
    depth_image[depth_image <= min_depth] = 0


    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=rgb_image.shape[1],
        height= rgb_image.shape[0],
        fx=camera_intrinsics[0, 0],
        fy=camera_intrinsics[1, 1],
        cx=camera_intrinsics[0, 2],
        cy=camera_intrinsics[1, 2]
    )

    # Convert depth image to Open3D format
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

    # Create an Open3D RGBD image
    rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=rgb_o3d,
        depth=depth_o3d,
        depth_scale=1.0,  # Adjust based on your depth scale
        depth_trunc=10.0,  # Truncate depth values beyond this distance
        convert_rgb_to_intensity=False
    )

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics
    )

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)