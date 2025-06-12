import numpy as np
import cv2
import time
import os
import yaml

from wire_detector_platforms import WireDetectorCPU, WireDetectorGPU

with open('wire_detect_config.yaml', 'r') as file:
    detection_config = yaml.safe_load(file)

folder = '/root/test_data/'
target_timestamp = 1746650644465219840 # straight wire

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
depth = np.load(depth_folder + depth_image_path).astype(np.float32)
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

iterations = 100
wire_detector_cpu = WireDetectorCPU(detection_config, camera_intrinsics)
wire_detector_gpu = WireDetectorGPU(detection_config, camera_intrinsics)

print("Starting CPU detection...")
cpu_detection_time = 0
cpu_roi_time = 0
cpu_ransac_time = 0
cpu_total_time = 0
for i in range(iterations):
# Create segmentation mask
    total_start_time = time.perf_counter()
    start_time_cpu = time.perf_counter()
    wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = wire_detector_cpu.detect_wires_2d(img)
    end_time_cpu = time.perf_counter()
    cpu_detection_time += (end_time_cpu - start_time_cpu)

    start_time_cpu = time.perf_counter()
    regions_of_interest, roi_line_counts = wire_detector_cpu.find_regions_of_interest(depth, avg_angle, midpoint_dists_wrt_center)
    end_time_cpu = time.perf_counter()
    cpu_roi_time += (end_time_cpu - start_time_cpu)

    start_time_cpu = time.perf_counter()
    fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = wire_detector_cpu.ransac_on_rois(regions_of_interest, roi_line_counts, avg_angle, depth, viz_img=img)
    end_time_cpu = time.perf_counter()
    cpu_ransac_time += (end_time_cpu - start_time_cpu)
    cpu_total_time += (end_time_cpu - total_start_time)

avg_cpu_detection_time = cpu_detection_time / iterations
avg_cpu_roi_time = cpu_roi_time / iterations
avg_cpu_ransac_time = cpu_ransac_time / iterations
avg_cpu_total_time = cpu_total_time / iterations
print(f"Average CPU detection time: {avg_cpu_detection_time:.6f} seconds, {1 / avg_cpu_detection_time:.6f} Hz")
print(f"Average CPU regions of interest time: {avg_cpu_roi_time:.6f} seconds, {1 / avg_cpu_roi_time:.6f} Hz")
print(f"Average CPU RANSAC on ROIs time: {avg_cpu_ransac_time:.6f} seconds, {1 / avg_cpu_ransac_time:.6f} Hz")
print(f"Average CPU total time: {avg_cpu_total_time:.6f} seconds, {1 / avg_cpu_total_time:.6f} Hz")


print("Starting GPU detection...")
# Now for the GPU detection
gpu_detection_time = 0
gpu_roi_time = 0
gpu_ransac_time = 0
gpu_total_time = 0
for i in range(iterations):
    total_start_time = time.perf_counter()
    start_time_gpu = time.perf_counter()
    wire_lines_gpu, wire_midpoints_gpu, avg_angle_gpu, midpoint_dists_wrt_center_gpu = wire_detector_gpu.detect_wires_2d(img)
    end_time_gpu = time.perf_counter()
    gpu_detection_time += (end_time_gpu - start_time_gpu)

    start_time_gpu = time.perf_counter()
    regions_of_interest_gpu, roi_line_counts_gpu = wire_detector_gpu.find_regions_of_interest(depth, avg_angle_gpu, midpoint_dists_wrt_center_gpu)
    end_time_gpu = time.perf_counter()
    gpu_roi_time += (end_time_gpu - start_time_gpu)

    start_time_gpu = time.perf_counter()
    fitted_lines_gpu, line_inlier_counts_gpu, roi_pcs_gpu, roi_point_colors_gpu, rgb_masked_gpu = wire_detector_gpu.ransac_on_rois(regions_of_interest_gpu, roi_line_counts_gpu, avg_angle_gpu, depth, viz_img=img)
    end_time_gpu = time.perf_counter()
    gpu_ransac_time += (end_time_gpu - start_time_gpu)
    gpu_total_time += (end_time_gpu - total_start_time)

avg_gpu_detection_time = gpu_detection_time / iterations
avg_gpu_roi_time = gpu_roi_time / iterations
avg_gpu_ransac_time = gpu_ransac_time / iterations
avg_gpu_total_time = gpu_total_time / iterations
print(f"Average GPU detection time: {avg_gpu_detection_time:.6f} seconds, {1 / avg_gpu_detection_time:.6f} Hz")
print(f"Average GPU regions of interest time: {avg_gpu_roi_time:.6f} seconds, {1 / avg_gpu_roi_time:.6f} Hz")
print(f"Average GPU RANSAC on ROIs time: {avg_gpu_ransac_time:.6f} seconds, {1 / avg_gpu_ransac_time:.6f} Hz")
print(f"Average GPU total time: {avg_gpu_total_time:.6f} seconds, {1 / avg_gpu_total_time:.6f} Hz")
print(f"Speedup: {avg_cpu_total_time / avg_gpu_total_time:.2f}x")



