import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
import yaml
import importlib

from wire_detector_platforms import WireDetectorCPU
# from ransac_bindings import ransac_on_rois
import viz_utils as vu

with open('wire_detection_python/wire_detect_config.yaml', 'r') as file:
    detection_config = yaml.safe_load(file)

folder = '/media/airlab/hummingbird/250815_vtolwire_2/'
target_timestamp = 1755288207710306880

rgb_folder = folder + 'rgb_images/'
depth_folder = folder + 'depth_images/'
camera_intrinsics_file = folder + 'rgb_camera_intrinsics.npy'
camera_intrinsics = np.load(camera_intrinsics_file)

closest_rgb_timestamp = None
closest_depth_timestamp = None
for image_name in os.listdir(rgb_folder):
    if image_name.endswith('.png'):
        timestamp = int(image_name.split('.')[0])
        if closest_rgb_timestamp is None or abs(timestamp - target_timestamp) < abs(closest_rgb_timestamp - target_timestamp):
            closest_rgb_timestamp = timestamp
            closest_rgb_timestamp_s = timestamp * 1e-9  # Convert to seconds
for image_name in os.listdir(depth_folder):
    if image_name.endswith('.npy'):
        timestamp = int(image_name.split('.')[0])
        if closest_depth_timestamp is None or abs(timestamp - target_timestamp) < abs(closest_depth_timestamp - target_timestamp):
            closest_depth_timestamp = timestamp
            closest_depth_timestamp_s = timestamp * 1e-9  # Convert to seconds

rgb_image_path = str(closest_rgb_timestamp) + '.png'
depth_image_path = str(closest_depth_timestamp) + '.npy'

print(f"Time difference between closest RGB and depth images: {abs(closest_rgb_timestamp_s - closest_depth_timestamp_s)}")
    
input_image_size = [480, 270] 
img = cv2.imread(rgb_folder + rgb_image_path)
depth = np.load(depth_folder + depth_image_path)
img = cv2.resize(img, (input_image_size[0], input_image_size[1]))
depth = cv2.resize(depth, (input_image_size[0], input_image_size[1]))
assert img is not None, "Image not found"
assert depth is not None, "Depth image not found"

original_image_size = img.shape[:2][::-1]  # (width, height)

# resize the camera intrinsics to match the input image size
camera_intrinsics[0, 0] *= input_image_size[0] / original_image_size[0]
camera_intrinsics[1, 1] *= input_image_size[1] / original_image_size[1]
camera_intrinsics[0, 2] *= input_image_size[0] / original_image_size[0]
camera_intrinsics[1, 2] *= input_image_size[1] / original_image_size[1]

wire_detector = WireDetectorCPU(detection_config, camera_intrinsics)

# Number of iterations for benchmarking
num_iterations = 100
warmup_iterations = 5

print(f"Running {warmup_iterations} warmup iterations...")
# Warmup iterations to stabilize timing
for _ in range(warmup_iterations):
    wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = wire_detector.detect_wires_2d(img)
    if avg_angle is not None:
        regions_of_interest, roi_line_counts = wire_detector.find_regions_of_interest(depth, avg_angle, midpoint_dists_wrt_center)
        if len(regions_of_interest) > 0:
            fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = wire_detector.ransac_on_rois_cpp(
                regions_of_interest, roi_line_counts, avg_angle, depth, viz_img=None)

print(f"Running {num_iterations} benchmark iterations...")
timings = []

for i in range(num_iterations):
    start_time = time.perf_counter()
    
    # Run full 3D wire detection pipeline
    wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = wire_detector.detect_wires_2d(img)
    
    if avg_angle is not None:
        regions_of_interest, roi_line_counts = wire_detector.find_regions_of_interest(depth, avg_angle, midpoint_dists_wrt_center)
        
        if len(regions_of_interest) > 0:
            fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = wire_detector.ransac_on_rois_cpp(
                regions_of_interest, roi_line_counts, avg_angle, depth, viz_img=None)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    timings.append(elapsed_time)
    
    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/{num_iterations} iterations...")

# Calculate statistics
timings = np.array(timings)
mean_time = np.mean(timings)
min_time = np.min(timings)
max_time = np.max(timings)
std_time = np.std(timings)
median_time = np.median(timings)

mean_fps = 1.0 / mean_time
min_fps = 1.0 / max_time  # Min FPS corresponds to max time
max_fps = 1.0 / min_time  # Max FPS corresponds to min time

print("\n" + "="*60)
print("3D Wire Detection Inference Speed Results")
print("="*60)
print(f"Number of iterations: {num_iterations}")
print(f"Warmup iterations: {warmup_iterations}")
print("\nTiming Statistics (seconds):")
print(f"  Mean:   {mean_time*1000:.3f} ms ({mean_time:.6f} s)")
print(f"  Median: {median_time*1000:.3f} ms ({median_time:.6f} s)")
print(f"  Min:    {min_time*1000:.3f} ms ({min_time:.6f} s)")
print(f"  Max:    {max_time*1000:.3f} ms ({max_time:.6f} s)")
print(f"  Std:    {std_time*1000:.3f} ms ({std_time:.6f} s)")
print("\nFrame Rate Statistics (FPS):")
print(f"  Mean:   {mean_fps:.2f} FPS")
print(f"  Min:    {min_fps:.2f} FPS")
print(f"  Max:    {max_fps:.2f} FPS")
print("="*60)

