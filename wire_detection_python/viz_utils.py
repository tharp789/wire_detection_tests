import numpy as np
import cv2
import open3d as o3d
import os
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
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
        norm_colors = np.array(colors) / 255.0  # Normalize colors to [0, 1]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(norm_colors)
        geometries.append(point_cloud)

    o3d.visualization.draw_geometries(geometries, 
                                      point_show_normal=True,
                                      mesh_show_back_face=True)

def create_renderer(width=1920, height=1080):
    renderer = OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # White background
    renderer.scene.scene.enable_sun_light(True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_point_path = os.path.join(script_dir, "viz_point.json")
    cam_params = o3d.io.read_pinhole_camera_parameters(viz_point_path)
    renderer.setup_camera(cam_params.intrinsic, cam_params.extrinsic)

    material = MaterialRecord()
    material.shader = "defaultUnlit"
    # material.shader = "defaultLit"
    
    return renderer, material

def capture_fitted_lines_in_image(renderer, material, img_name, fitted_lines, roi_pcs, roi_point_colors):
    # Add cylinders for lines
    for i, (start, end) in enumerate(fitted_lines):
        cyl = create_cylinder_between_points(start, end, radius=0.075)
        renderer.scene.add_geometry(f"cyl_{i}", cyl, material)

    # Add point clouds
    for i, (points, colors) in enumerate(zip(roi_pcs, roi_point_colors)):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        # Normalize colors to [0, 1] range for Open3D (colors come in as uint8 [0, 255])
        if colors is not None and len(colors) > 0:
            norm_colors = np.array(colors, dtype=np.float32)
            # If colors are in [0, 255] range, normalize them
            if norm_colors.max() > 1.0:
                norm_colors = norm_colors / 255.0
            pc.colors = o3d.utility.Vector3dVector(norm_colors)
        renderer.scene.add_geometry(f"pc_{i}", pc, material)

    # Render to image
    img_o3d = renderer.render_to_image()
    o3d.io.write_image(img_name, img_o3d, 9)
    renderer.scene.clear_geometry()

def depth_pc_in_image(renderer, material, img_name, depth_image, rgb_image, intrinsics):

    o3d_depth = o3d.geometry.Image(depth_image * 1000)  # Scale depth for visualization
    o3d_rgb = o3d.geometry.Image(rgb_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d_rgb,
        depth=o3d_depth,
        convert_rgb_to_intensity=False,
        depth_scale=1000.0,
        depth_trunc=3.0  # Truncate far-away points
    )
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=rgb_image.shape[1],
        height=rgb_image.shape[0],
        fx=intrinsics[0, 0],
        fy=intrinsics[1, 1],
        cx=intrinsics[0, 2],
        cy=intrinsics[1, 2]
    )
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("pc", pc, material)

    # Render to image
    img_o3d = renderer.render_to_image()
    o3d.io.write_image(img_name, img_o3d, 9)
    renderer.scene.clear_geometry()
    
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

def extend_line_to_edges(p1, p2, width, height):
    """Extend a line segment to the edges of the image."""
    x1, y1 = p1
    x2, y2 = p2
    
    # Calculate line equation: y = mx + b
    if abs(x2 - x1) < 1e-6:  # Vertical line
        return (int(x1), 0), (int(x1), height - 1)
    
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    
    # Find intersections with image boundaries
    intersections = []
    
    # Left edge (x = 0)
    y_left = m * 0 + b
    if 0 <= y_left < height:
        intersections.append((0, int(y_left)))
    
    # Right edge (x = width - 1)
    y_right = m * (width - 1) + b
    if 0 <= y_right < height:
        intersections.append((width - 1, int(y_right)))
    
    # Top edge (y = 0)
    if abs(m) > 1e-6:
        x_top = (0 - b) / m
        if 0 <= x_top < width:
            intersections.append((int(x_top), 0))
    
    # Bottom edge (y = height - 1)
    if abs(m) > 1e-6:
        x_bottom = ((height - 1) - b) / m
        if 0 <= x_bottom < width:
            intersections.append((int(x_bottom), height - 1))
    
    # If we have at least 2 intersections, use them; otherwise use original points
    if len(intersections) >= 2:
        # Use the two intersections that are furthest apart
        max_dist = 0
        best_pair = (intersections[0], intersections[1])
        for i in range(len(intersections)):
            for j in range(i + 1, len(intersections)):
                dist = np.sqrt((intersections[i][0] - intersections[j][0])**2 + 
                                (intersections[i][1] - intersections[j][1])**2)
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (intersections[i], intersections[j])
        return best_pair
    elif len(intersections) == 1:
        # One intersection, extend from it through the original line
        return intersections[0], (int(x2), int(y2))
    else:
        # No valid intersections, use original points
        return (int(x1), int(y1)), (int(x2), int(y2))

def draw_3d_lines_on_image(image, lines, camera_intrinsics, color=(255, 0, 0), thickness=2):
    """
    Draws 3D lines on a 2D image, extended to the edges of the image.
    
    Parameters:
    - image: The image on which to draw the lines.
    - lines: List of 3D lines, each as ((x1, y1, z1), (x2, y2, z2)).
    - camera_intrinsics: Camera intrinsic matrix.
    - color: Color of the line in BGR format (default: red).
    - thickness: Thickness of the line.
    """
    original_size = image.shape[:2]
    image = cv2.resize(image, (1280, 720))  # Resize to 1280x720 for better visualization
    scale = np.array([image.shape[1] / original_size[1], image.shape[0] / original_size[0]])
    img_height, img_width = image.shape[:2]

    for line in lines:
        start_2d = np.dot(camera_intrinsics, np.array([line[0][0], line[0][1], line[0][2]]))
        end_2d = np.dot(camera_intrinsics, np.array([line[1][0], line[1][1], line[1][2]]))
        start_2d = (start_2d[:2] / start_2d[2])
        end_2d = (end_2d[:2] / end_2d[2])
        scaled_start_2d = start_2d * scale
        scaled_end_2d = end_2d * scale
        
        # Extend line to image edges
        p1, p2 = extend_line_to_edges(scaled_start_2d, scaled_end_2d, img_width, img_height)
        cv2.line(image, p1, p2, color, thickness)

    return image

def get_3d_lines_in_pixels(img, lines, camera_intrinsics):
    """
    Draws 3D lines on a 2D image, extended to the edges of the image.
    
    Parameters:
    - lines: List of 3D lines, each as ((x1, y1, z1), (x2, y2, z2)).
    - camera_intrinsics: Camera intrinsic matrix.
    """
    img_height, img_width = img.shape[:2]

    line_list = []
    for line in lines:
        start_2d = np.dot(camera_intrinsics, np.array([line[0][0], line[0][1], line[0][2]]))
        end_2d = np.dot(camera_intrinsics, np.array([line[1][0], line[1][1], line[1][2]]))
        start_2d = (start_2d[:2] / start_2d[2])
        end_2d = (end_2d[:2] / end_2d[2])
        scaled_start_2d = start_2d
        scaled_end_2d = end_2d
        
        # Extend line to image edges
        p1, p2 = extend_line_to_edges(scaled_start_2d, scaled_end_2d, img_width, img_height)
        line_list.append((p1, p2))

    return line_list

def make_video(image_files, output_path, fps=10):
    if not image_files:
        return
    # Read the first image to get frame size
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)

        if frame is not None:
            print(f"Writing frame {i+1}/{len(image_files)}: {img_path}")
            video_writer.write(frame)
    video_writer.release()

def get_bounding_box_2d_from_3d_line(line_2d, bb_width, image_shape: tuple, return_polygon=False, padding=0):
    """
    Computes a 2D bounding box around a 3D line projected into 2D image space.
    
    Parameters:
    - line_2d: Tuple of two points representing the 2D projection of the 3D line ((x1, y1), (x2, y2)).
    - bb_width: Width of the bounding box in pixels.
    - image_shape: Tuple of (height, width) of the image.
    - return_polygon: If True, returns polygon corners as list of (x, y) tuples. 
                     If False, returns bounding box as (x_min, y_min, x_max, y_max).
    - padding: Additional padding in pixels to add to the bounding box width (default: 0).
    
    Returns:
    - If return_polygon=True: List of 4 corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    - If return_polygon=False: Tuple representing the bounding box (x_min, y_min, x_max, y_max).
    """
    # Apply padding to the bounding box width
    effective_width = bb_width + padding
    
    p1, p2 = line_2d
    direction = np.array(p2) - np.array(p1)
    length = np.linalg.norm(direction)
    if length == 0:
        if return_polygon:
            half_w = effective_width / 2
            return [
                (int(p1[0] - half_w), int(p1[1] - half_w)),
                (int(p1[0] + half_w), int(p1[1] - half_w)),
                (int(p1[0] + half_w), int(p1[1] + half_w)),
                (int(p1[0] - half_w), int(p1[1] + half_w))
            ]
        else:
            return (int(p1[0] - effective_width / 2), int(p1[1] - effective_width / 2), int(p1[0] + effective_width / 2), int(p1[1] + effective_width / 2))
    direction = direction / length

    # Compute perpendicular direction
    perp_direction = np.array([-direction[1], direction[0]])

    # Compute corners of the bounding box with padding
    corner1 = np.array(p1) + (effective_width / 2) * perp_direction
    corner2 = np.array(p1) - (effective_width / 2) * perp_direction
    corner3 = np.array(p2) + (effective_width / 2) * perp_direction
    corner4 = np.array(p2) - (effective_width / 2) * perp_direction

    corner1_norm = np.clip(corner1, [0, 0], [image_shape[1] - 1, image_shape[0] - 1])
    corner2_norm = np.clip(corner2, [0, 0], [image_shape[1] - 1, image_shape[0] - 1])
    corner3_norm = np.clip(corner3, [0, 0], [image_shape[1] - 1, image_shape[0] - 1])
    corner4_norm = np.clip(corner4, [0, 0], [image_shape[1] - 1, image_shape[0] - 1])

    if return_polygon:
        # Return polygon corners as list of (x, y) tuples
        # Order: corner1 (p1 + perp), corner2 (p1 - perp), corner4 (p2 - perp), corner3 (p2 + perp)
        # This forms a quadrilateral going around the line
        corners = [
            (int(corner1_norm[0]), int(corner1_norm[1])),  # p1 + perp
            (int(corner2_norm[0]), int(corner2_norm[1])),  # p1 - perp
            (int(corner4_norm[0]), int(corner4_norm[1])),  # p2 - perp
            (int(corner3_norm[0]), int(corner3_norm[1]))   # p2 + perp
        ]
        return corners
    else:
        return (int(min(corner1_norm[0], corner2_norm[0], corner3_norm[0], corner4_norm[0])),
                int(min(corner1_norm[1], corner2_norm[1], corner3_norm[1], corner4_norm[1])),
                int(max(corner1_norm[0], corner2_norm[0], corner3_norm[0], corner4_norm[0])),
                int(max(corner1_norm[1], corner2_norm[1], corner3_norm[1], corner4_norm[1])))

# Create RGB mask visualization showing ROI and non-ROI regions
def create_roi_visualization_mask(img, rois, non_rois, perp_angle, roi_color=(0, 255, 0), non_roi_color=(128, 128, 128), alpha=0.3):
    """
    Create a visualization mask showing regions of interest and non-interest.
    
    Args:
        img: Input RGB image (BGR format)
        rois: List of (start, end) tuples for regions of interest
        non_rois: List of (start, end) tuples for regions of no interest
        avg_angle: Average angle of wires in radians
        roi_color: BGR color for ROI regions (default: green)
        non_roi_color: BGR color for non-ROI regions (default: gray)
        alpha: Transparency factor for overlay (0.0 to 1.0)
    
    Returns:
        Visualization image with colored regions
    """
    img_shape = img.shape[:2]
    img_center = (img_shape[1] * 0.5, img_shape[0] * 0.5)  # (x, y)
    
    # Create overlay image
    overlay = img.copy()
    
    # Draw ROI regions in green
    for start, end in rois:
        center_dist = 0.5 * (start + end)
        length = abs(end - start)
        
        # Compute center offset in image coordinates
        dx = center_dist * np.cos(perp_angle)
        dy = center_dist * np.sin(perp_angle)
        center_coords = (img_center[0] + dx, img_center[1] + dy)
        
        # Define rectangle size: length along projected axis, large height perpendicular
        size = (length, img_shape[0] * 2)  # (width, height)
        
        # Create rotated rectangle
        rect = (center_coords, size, np.degrees(perp_angle))
        box = cv2.boxPoints(rect).astype(int)
        
        # Draw filled polygon with ROI color
        cv2.fillConvexPoly(overlay, box, roi_color)
    
    # Draw non-ROI regions in gray/red
    for start, end in non_rois:
        center_dist = 0.5 * (start + end)
        length = abs(end - start)
        
        # Compute center offset in image coordinates
        dx = center_dist * np.cos(perp_angle)
        dy = center_dist * np.sin(perp_angle)
        center_coords = (img_center[0] + dx, img_center[1] + dy)
        
        # Define rectangle size
        size = (length, img_shape[0] * 2)
        
        # Create rotated rectangle
        rect = (center_coords, size, np.degrees(perp_angle))
        box = cv2.boxPoints(rect).astype(int)
        
        # Draw filled polygon with non-ROI color
        cv2.fillConvexPoly(overlay, box, non_roi_color)
    
    # Blend overlay with original image
    result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    return result