import numpy as np
import cv2
import time
import os
import sys
import yaml

import open3d as o3d

from wire_detector_platforms import WireDetectorCPU
import viz_utils as vu

DO_RENDERING = True
OVERWRITE = False

script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of the script
config_path = os.path.join(script_dir, "wire_detect_config.yaml")

with open(config_path, 'r') as file:
    detection_config = yaml.safe_load(file)

# Create a WireDetector instance
input_image_size = [480, 270]

folder = "/media/tyler/Storage/field_tests/250815_vtolwire_2/" 

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

ransac_results_2d = folder + 'ransac_results_2d/'
ransac_results_3d = folder + 'ransac_results_3d/'

if os.path.exists(ransac_results_2d) == False:
    os.makedirs(ransac_results_2d)
if os.path.exists(ransac_results_3d) == False:
    os.makedirs(ransac_results_3d)

avg_frequency = 0.0
renderer, material = vu.create_renderer()

if OVERWRITE == False:
    current_2d_results = os.listdir(ransac_results_2d)

if DO_RENDERING:
    sync_ransac_time = 0.0
    async_ransac_time = 0.0
    roi_time = 0.0
    detect_wire_time = 0.0
    for file in rgb_file_list:
        if OVERWRITE == False and f"{int(file.split('.')[0])}_2d.png" in current_2d_results:
            continue
        start_time = time.perf_counter()
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

        start_time = time.perf_counter()
        start_detect_wire = time.perf_counter()
        wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = wire_detector.detect_wires_2d(rgb_img)
        detect_wire_time += time.perf_counter() - start_detect_wire

        fitted_lines = None
        if wire_lines is not None and len(wire_lines) > 0:
            start_roi = time.perf_counter()
            regions_of_interest, roi_line_counts = wire_detector.find_regions_of_interest(depth_img, avg_angle, midpoint_dists_wrt_center)
            roi_time += time.perf_counter() - start_roi

            start_ransac_sync = time.perf_counter()
            fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = wire_detector.ransac_on_rois(regions_of_interest, roi_line_counts, avg_angle, depth_img, viz_img=rgb_img)
            sync_ransac_time += time.perf_counter() - start_ransac_sync

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            avg_frequency += 1.0 / elapsed_time

            rgb_masked = vu.draw_3d_line_on_image(rgb_masked, fitted_lines, camera_intrinsics, color=(0, 255, 0), thickness=3)

        rgb_img_path = os.path.join(ransac_results_2d, str(rgb_timestamp) + '_2d.png')
        pc_img_path = os.path.join(ransac_results_3d, str(rgb_timestamp) + '_3d.png')
        if fitted_lines is None or len(fitted_lines) == 0:
            large_rgb_img = cv2.resize(rgb_img, (1920, 1080))
            cv2.imwrite(rgb_img_path, large_rgb_img)
            vu.depth_pc_in_image(renderer, material, pc_img_path, depth_img, rgb_img, camera_intrinsics)
        else:
            vu.capture_fitted_lines_in_image(renderer, material, pc_img_path, fitted_lines, roi_pcs, roi_point_colors)
            cv2.imwrite(rgb_img_path, rgb_masked)

    avg_frequency /= len(rgb_file_list)
    avg_sync_ransac_time = sync_ransac_time / len(rgb_file_list)
    avg_roi_time = roi_time / len(rgb_file_list)
    avg_detect_wire_time = detect_wire_time / len(rgb_file_list)
    print(f"Detect 3D Average frequency: {avg_frequency:.4f}, Average Period: {1/avg_frequency:.4f} seconds")
    print(f"Average 2D Detect Wire frequency: {1/avg_detect_wire_time:.4f}, Average Period: {avg_detect_wire_time:.4f} seconds")
    print(f"Average ROI frequency: {1/avg_roi_time:.4f}, Average Period: {avg_roi_time:.4f} seconds")
    print(f"Average RANSAC frequency: {1/avg_sync_ransac_time:.4f}, Average Period: {avg_sync_ransac_time:.4f} seconds")

# copy the ransac results into a seperate 2d and 3d folder
video_2d_name = folder + 'ransac_results_2d.mp4'
video_3d_name = folder + 'ransac_results_3d.mp4'

frames_2d = sorted(
    [os.path.join(ransac_results_2d, f) for f in os.listdir(ransac_results_2d) if f.endswith(".png")]
)

frames_3d = sorted(
    [os.path.join(ransac_results_3d, f) for f in os.listdir(ransac_results_3d) if f.endswith(".png")]
)

vu.make_video(frames_2d, video_2d_name, fps=30)
vu.make_video(frames_3d, video_3d_name, fps=30)