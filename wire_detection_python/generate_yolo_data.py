import numpy as np
import cv2
import time
import os
import sys
import yaml

import open3d as o3d

from wire_detector_platforms import WireDetectorCPU
import viz_utils as vu

OVERWRITE = False
MAX_WIRE_COUNT = 2
BB_PADDING = 10  # Padding in pixels to add to bounding box width

script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of the script
config_path = os.path.join(script_dir, "wire_detect_config.yaml")

with open(config_path, 'r') as file:
    detection_config = yaml.safe_load(file)

# Create a WireDetector instance
input_image_size = [480, 270]

folder = "/home/tyler/Documents/yolo_wire/raw_data/hawkins_2_19_1/" 

rgb_folder = folder + 'rgb_images/'
depth_folder = folder + 'depth_images/'
camera_intrinsics_file = folder + 'rgb_camera_intrinsics.npy'
camera_intrinsics = np.load(camera_intrinsics_file)
rgb_file_list = sorted([f for f in os.listdir(rgb_folder) if f.endswith('.png')])
depth_file_list = sorted([f for f in os.listdir(depth_folder) if f.endswith('.npy')])

if len(rgb_file_list) == 0:
    raise ValueError(f"No PNG files found in {rgb_folder}")

# Get original image size from first RGB image
first_img = cv2.imread(os.path.join(rgb_folder, rgb_file_list[0]))
if first_img is None:
    raise ValueError(f"Could not read first image: {rgb_file_list[0]}")
original_image_size = first_img.shape[:2][::-1]  # (width, height)

# resize the camera intrinsics to match the input image size
camera_intrinsics[0, 0] *= input_image_size[0] / original_image_size[0]
camera_intrinsics[1, 1] *= input_image_size[1] / original_image_size[1]
camera_intrinsics[0, 2] *= input_image_size[0] / original_image_size[0]
camera_intrinsics[1, 2] *= input_image_size[1] / original_image_size[1]

wire_detector = WireDetectorCPU(detection_config, camera_intrinsics)

yolo_generation_folder = folder + 'yolo_generation/'
yolo_image_folder = yolo_generation_folder + 'images/'
bb_viz_folder = yolo_generation_folder + 'bb_viz/'
yolo_label_folder = yolo_generation_folder + 'labels/'

if not os.path.exists(yolo_generation_folder):
    os.makedirs(yolo_generation_folder)
if not os.path.exists(yolo_image_folder):
    os.makedirs(yolo_image_folder)
if not os.path.exists(bb_viz_folder):
    os.makedirs(bb_viz_folder)
if not os.path.exists(yolo_label_folder):
    os.makedirs(yolo_label_folder)

if OVERWRITE == False:
    current_2d_results = os.listdir(yolo_image_folder) if os.path.exists(yolo_image_folder) else []

for i, file in enumerate(rgb_file_list):

    print(f"Processing file {i} of {len(rgb_file_list)}: {file}")
    rgb_timestamp = int(file.split('.')[0])
    if OVERWRITE == False and f"{rgb_timestamp}.png" in current_2d_results:
        continue
    
    closest_depth_timestamp = None
    closest_depth_file = None
    for depth_file in depth_file_list:
        depth_timestamp = int(depth_file.split('.')[0])
        if closest_depth_timestamp is None or abs(depth_timestamp - rgb_timestamp) < abs(closest_depth_timestamp - rgb_timestamp):
            closest_depth_timestamp = depth_timestamp
            closest_depth_file = os.path.join(depth_folder, depth_file)
    
    if closest_depth_file is None:
        print(f"Warning: No depth file found for RGB timestamp {rgb_timestamp}, skipping...")
        continue
            
    rgb_img = cv2.imread(os.path.join(rgb_folder, file))
    if rgb_img is None:
        print(f"Warning: Could not read RGB image {file}, skipping...")
        continue
    
    depth_img = np.load(closest_depth_file)
    rgb_img = cv2.resize(rgb_img, (input_image_size[0], input_image_size[1]))
    depth_img = cv2.resize(depth_img, (input_image_size[0], input_image_size[1]))
    
    min_depth = 0.5
    depth_img[depth_img <= min_depth] = 0

    wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = wire_detector.detect_wires_2d(rgb_img)

    fitted_lines = None
    if wire_lines is not None and len(wire_lines) > 0:
        bb_viz_img = rgb_img.copy()
        regions_of_interest, roi_line_counts = wire_detector.find_regions_of_interest(depth_img, avg_angle, midpoint_dists_wrt_center)

        # Use C++ implementation for better performance
        fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = wire_detector.ransac_on_rois_cpp(
            regions_of_interest, roi_line_counts, avg_angle, depth_img, viz_img=rgb_img
        )

        # Skip images with no wires detected
        if fitted_lines is None or len(fitted_lines) == 0:
            print(f"Skipping frame {rgb_timestamp}: No wires detected after RANSAC")
            continue

        # Skip images with more wires than MAX_WIRE_COUNT
        if len(fitted_lines) > MAX_WIRE_COUNT:
            print(f"Skipping frame {rgb_timestamp}: {len(fitted_lines)} wires detected (max: {MAX_WIRE_COUNT})")
            continue

        # Get 2D pixel coordinates of 3D lines (use original image, not masked)
        lines = vu.get_3d_lines_in_pixels(rgb_img, fitted_lines, camera_intrinsics)

        yolo_label_path = os.path.join(yolo_label_folder, str(rgb_timestamp) + '.txt')
        # Clear existing label file if overwriting
        if OVERWRITE or not os.path.exists(yolo_label_path):
            open(yolo_label_path, 'w').close()
        
        img_height, img_width = rgb_img.shape[:2]
        
        for i, line_3d in enumerate(fitted_lines):
            line_depth = (line_3d[0][2] + line_3d[1][2]) / 2.0
            # Calculate bounding box width based on wire diameter (3.81 cm = 0.0381 m)
            avg_focal_length = (camera_intrinsics[0, 0] + camera_intrinsics[1, 1]) / 2.0
            bb_width = (avg_focal_length * 0.0381) / line_depth  # 3.81 cm wire diameter
            bb = vu.get_bounding_box_2d_from_3d_line(lines[i], bb_width, image_shape=rgb_img.shape, return_polygon=True, padding=BB_PADDING)

            # Normalize coordinates for YOLO format (0-1 range)
            normalized_coords = []
            for x, y in bb:
                norm_x = x / img_width
                norm_y = y / img_height
                normalized_coords.extend([norm_x, norm_y])
            
            # Write YOLO format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (all normalized)
            with open(yolo_label_path, 'a') as label_file:
                coords_str = ' '.join([f'{coord:.6f}' for coord in normalized_coords])
                label_file.write(f"0 {coords_str}\n")  # Class 0 for wire
            
            # Draw bounding box for visualization
            bb_viz_img = cv2.polylines(bb_viz_img, [np.array(bb, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

        bb_viz_path = os.path.join(bb_viz_folder, str(rgb_timestamp) + '.png')
        cv2.imwrite(bb_viz_path, bb_viz_img)

        yolo_image_path = os.path.join(yolo_image_folder, str(rgb_timestamp) + '.png')
        rgb_img = cv2.resize(rgb_img, (1280, 720))
        cv2.imwrite(yolo_image_path, rgb_img)

        print(f"Processed frame {rgb_timestamp} with {len(fitted_lines)} wires detected.")
