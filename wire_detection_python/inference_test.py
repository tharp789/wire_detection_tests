import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
import yaml
import importlib
import psutil

from wire_detector_platforms import WireDetectorCPU
# from ransac_bindings import ransac_on_rois
import viz_utils as vu

with open('wire_detection_python/wire_detect_config.yaml', 'r') as file:
    detection_config = yaml.safe_load(file)

folder = '/media/tyler/hummingbird/250815_vtolwire_2/'
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
cpu_percentages = []
cpu_frequencies = []
memory_usages = []
process = psutil.Process()

# Get initial CPU state
initial_cpu_percent = psutil.cpu_percent(interval=None)
initial_cpu_freq = psutil.cpu_freq()
initial_memory = process.memory_info().rss / 1024 / 1024  # MB

for i in range(num_iterations):
    start_time = time.perf_counter()
    
    # Get CPU metrics before inference
    cpu_percent_before = psutil.cpu_percent(interval=None)
    cpu_freq_before = psutil.cpu_freq().current if psutil.cpu_freq() else None
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
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
    
    # Get CPU metrics after inference
    cpu_percent_after = psutil.cpu_percent(interval=None)
    cpu_freq_after = psutil.cpu_freq().current if psutil.cpu_freq() else None
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Store metrics (use average of before/after for CPU percent, or process-specific)
    process_cpu_percent = process.cpu_percent(interval=None)
    cpu_percentages.append(process_cpu_percent if process_cpu_percent > 0 else cpu_percent_after)
    if cpu_freq_after is not None:
        cpu_frequencies.append(cpu_freq_after)
    memory_usages.append(memory_after)
    
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

# Calculate CPU statistics
cpu_percentages = np.array(cpu_percentages)
cpu_mean = np.mean(cpu_percentages)
cpu_max = np.max(cpu_percentages)
cpu_min = np.min(cpu_percentages)
cpu_std = np.std(cpu_percentages)

cpu_frequencies = np.array(cpu_frequencies) if len(cpu_frequencies) > 0 else np.array([])
if len(cpu_frequencies) > 0:
    freq_mean = np.mean(cpu_frequencies)
    freq_max = np.max(cpu_frequencies)
    freq_min = np.min(cpu_frequencies)
    freq_std = np.std(cpu_frequencies)

memory_usages = np.array(memory_usages)
memory_mean = np.mean(memory_usages)
memory_max = np.max(memory_usages)
memory_min = np.min(memory_usages)
memory_std = np.std(memory_usages)

# Get system-wide CPU info
cpu_count = psutil.cpu_count(logical=True)
cpu_count_physical = psutil.cpu_count(logical=False)
cpu_percent_system = psutil.cpu_percent(interval=1.0)

# Get per-core CPU usage
cpu_percent_per_core = psutil.cpu_percent(interval=1.0, percpu=True)

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
print("\n" + "="*60)
print("CPU Benchmarking Results")
print("="*60)
print(f"CPU Information:")
print(f"  Physical cores: {cpu_count_physical}")
print(f"  Logical cores:  {cpu_count}")
if len(cpu_frequencies) > 0:
    print(f"  Frequency:      {freq_mean:.2f} MHz (avg), {freq_min:.2f}-{freq_max:.2f} MHz (range)")
print(f"\nProcess CPU Usage:")
print(f"  Mean:   {cpu_mean:.2f}%")
print(f"  Min:    {cpu_min:.2f}%")
print(f"  Max:    {cpu_max:.2f}%")
print(f"  Std:    {cpu_std:.2f}%")
print(f"\nSystem CPU Usage (current): {cpu_percent_system:.2f}%")
print(f"Per-core CPU Usage:")
for i, core_usage in enumerate(cpu_percent_per_core):
    print(f"  Core {i}: {core_usage:.2f}%")
print("\nMemory Usage:")
print(f"  Mean:   {memory_mean:.2f} MB")
print(f"  Min:    {memory_min:.2f} MB")
print(f"  Max:    {memory_max:.2f} MB")
print(f"  Std:    {memory_std:.2f} MB")
print(f"  Initial: {initial_memory:.2f} MB")
print(f"  Peak increase: {memory_max - initial_memory:.2f} MB")
print("="*60)

