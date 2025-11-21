import numpy as np
import cv2
import time
import os
import yaml

from wire_detector_platforms import WireDetectorCPU
import viz_utils as vu

GT_Z_DISTANCE_M = 5.34

def fold_angles_from_0_to_pi(angles):
    '''
    Fold angles to the range [0, π].
    '''
    angles = np.asarray(angles)  # Ensure input is an array
    angles = angles % (2 * np.pi)  # Wrap into [0, 2π)

    # Fold anything > π into [0, π]
    folded = np.where(angles > np.pi, angles - np.pi, angles)

    return folded.item() if np.isscalar(angles) else folded

def calculate_pitch_and_yaw(line):
    # line is a tuple of two 3D points: ((x0, y0, z0), (x1, y1, z1))
    point0 = np.array(line[0])  # (x0, y0, z0)
    point1 = np.array(line[1])  # (x1, y1, z1)
    
    # Calculate pitch and yaw from 3D line
    # Yaw: angle in XY plane (horizontal)
    yaw = np.arctan2(point1[1] - point0[1], point1[0] - point0[0])
    # Pitch: angle from horizontal plane (vertical)
    dx = point1[0] - point0[0]
    dy = point1[1] - point0[1]
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    pitch = np.arctan2(point1[2] - point0[2], horizontal_dist)
    
    pitch = fold_angles_from_0_to_pi(pitch)
    yaw = fold_angles_from_0_to_pi(yaw)
    return pitch, yaw

script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of the script
config_path = os.path.join(script_dir, "wire_detect_config.yaml")

with open(config_path, 'r') as file:
    detection_config = yaml.safe_load(file)

# Create a WireDetector instance
input_image_size = [480, 270]

folder = "/home/tyler/Documents/yolo_wire/raw_data/wire_data_20250828_104119/" 

rgb_folder = folder + 'rgb_images/'
depth_folder = folder + 'depth_images/'
camera_intrinsics_file = folder + 'rgb_camera_intrinsics.npy'
camera_intrinsics = np.load(camera_intrinsics_file)
rgb_file_list = sorted(os.listdir(rgb_folder))
depth_file_list = sorted(os.listdir(depth_folder))
original_image_size = cv2.imread(os.path.join(rgb_folder, rgb_file_list[0])).shape[:2][::-1]  # (width, height)

# resize the camera intrinsics to match the input image size
camera_intrinsics[0, 0] *= input_image_size[0] / original_image_size[0]
camera_intrinsics[1, 1] *= input_image_size[1] / original_image_size[1]
camera_intrinsics[0, 2] *= input_image_size[0] / original_image_size[0]
camera_intrinsics[1, 2] *= input_image_size[1] / original_image_size[1]

wire_detector = WireDetectorCPU(detection_config, camera_intrinsics)

# Collect all data across all frames
all_pitches = []
all_yaws = []
all_xs = []
all_ys = []
all_zs = []

for file in rgb_file_list:
    rgb_timestamp = int(file.split('.')[0])
    closest_depth_timestamp = None
    for depth_file in depth_file_list:
        if closest_depth_timestamp is None or abs(int(depth_file.split('.')[0]) - rgb_timestamp) < abs(int(closest_depth_timestamp.split('.')[0]) - rgb_timestamp):
            closest_depth_timestamp = depth_file
            closest_depth_file = os.path.join(depth_folder, str(closest_depth_timestamp))
            
    rgb_img = cv2.imread(os.path.join(rgb_folder, file))
    depth_img = np.load(closest_depth_file)
    rgb_img = cv2.resize(rgb_img, (input_image_size[0], input_image_size[1]))
    depth_img = cv2.resize(depth_img, (input_image_size[0], input_image_size[1]))
    
    min_depth = 0.5
    depth_img[depth_img <= min_depth] = 0

    wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = wire_detector.detect_wires_2d(rgb_img)

    if wire_lines is not None and len(wire_lines) > 0:
        regions_of_interest, roi_line_counts = wire_detector.find_regions_of_interest(depth_img, avg_angle, midpoint_dists_wrt_center)

        fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = wire_detector.ransac_on_rois(regions_of_interest, roi_line_counts, avg_angle, depth_img, viz_img=rgb_img)

        rgb_masked = vu.draw_3d_lines_on_image(rgb_masked, fitted_lines, camera_intrinsics, color=(0, 255, 0), thickness=3)

        for line in fitted_lines:
            pitch, yaw = calculate_pitch_and_yaw(line)
            all_pitches.append(pitch)
            all_yaws.append(yaw)
            all_xs.append(line[0][0])
            all_ys.append(line[0][1])
            all_zs.append(line[0][2])

# Calculate metrics after collecting all data
if len(all_zs) > 0:
    all_pitches = np.array(all_pitches)
    all_yaws = np.array(all_yaws)
    all_xs = np.array(all_xs)
    all_ys = np.array(all_ys)
    all_zs = np.array(all_zs)
    
    # Calculate metrics
    mean_vertical_distance = np.mean(all_zs)
    std_dev_vertical_distance = np.std(all_zs)
    
    # Horizontal error: distance from origin in XY plane
    x_std_dev = np.std(all_xs)
    y_std_dev = np.std(all_ys)
    horizontal_distances = np.sqrt(all_xs**2 + all_ys**2)
    std_dev_horizontal_error = np.std(horizontal_distances)
    
    # Ground truth vertical error: absolute error from ground truth
    ground_truth_vertical_error = np.mean(np.abs(all_zs - GT_Z_DISTANCE_M))
    
    # Convert angles to degrees for deviation
    yaw_deviation_deg = np.std(all_yaws) * 180 / np.pi
    pitch_deviation_deg = np.std(all_pitches) * 180 / np.pi
    
    # Extract folder name for output file
    folder_name = os.path.basename(os.path.normpath(folder))
    output_file = os.path.join(script_dir, f"metrics_{folder_name}.txt")
    
    # Output in LaTeX table format
    metrics_output = []
    metrics_output.append("="*60)
    metrics_output.append("METRICS TABLE (LaTeX format):")
    metrics_output.append("="*60)
    metrics_output.append(f"Mean Vertical Distance (m) & {mean_vertical_distance:.5f} \\\\")
    metrics_output.append(f"Std Dev Vertical Distance (m) & {std_dev_vertical_distance:.5f} \\\\")
    metrics_output.append(f"Std Dev X Error (m) & {x_std_dev:.5f} \\\\")
    metrics_output.append(f"Std Dev Y Error (m) & {y_std_dev:.5f} \\\\")
    metrics_output.append(f"Std Dev Horizontal Error (m) & {std_dev_horizontal_error:.5f} \\\\")
    metrics_output.append(f"Ground Truth Vertical Error (m) & {ground_truth_vertical_error:.5f} \\\\")
    metrics_output.append(f"Yaw Deviation (deg) & {yaw_deviation_deg:.5f} \\\\")
    metrics_output.append(f"Pitch Deviation (deg) & {pitch_deviation_deg:.5f} \\\\")
    metrics_output.append("="*60)
    
    # Print to console
    print("\n" + "\n".join(metrics_output))
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(metrics_output))
        f.write("\n")
    
    print(f"\nMetrics saved to: {output_file}")
else:
    print("No wires detected in any frame.")
