min_depth = 0.5
depth[depth <= min_depth] = 0

depth_viz = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
depth_viz = cv2.resize(depth_viz, (1280, 720))
cv2.imwrite('detect_3d_output/depth_viz.jpg', depth_viz)
plt.figure()
plt.imshow(depth_viz)

start_time = time.perf_counter()
depth_gradient_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=11)
depth_gradient_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=11)
perp_angle = wdu.perpendicular_angle_rad(avg_angle)
depth_gradient = depth_gradient_x * np.cos(perp_angle) + depth_gradient_y * np.sin(perp_angle)

distance, depth_gradient_1d = wdu.project_image_to_axis(depth_gradient, perp_angle)
# depth_gradient_1d = np.abs(depth_gradient_1d)
# depth_gradient_1d = depth_gradient_1d / np.max(depth_gradient_1d)

dist_hist, bin_edges = np.histogram(distance, bins=np.arange(np.min(distance), np.max(distance), 1), weights=depth_gradient_1d)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

negative_grads = dist_hist[dist_hist < 0]
negative_bin_centers = bin_centers[dist_hist < 0]
negative_grads = negative_grads / np.max(np.abs(negative_grads))  # Normalize negative gradients
negative_indices_masked = negative_grads < - wire_detector.grad_bin_avg_threshold
negative_diff = np.diff(negative_indices_masked.astype(int))
negative_diff = np.concatenate(([0], negative_diff))
negative_diff = np.where(negative_diff < 0, negative_diff, 0)  # Keep only negative differences

positive_grads = dist_hist[dist_hist > 0]
positive_bin_centers = bin_centers[dist_hist > 0]
positive_grads = positive_grads / np.max(positive_grads)  # Normalize positive gradients
positive_indices_masked = positive_grads > wire_detector.grad_bin_avg_threshold
positive_diff = np.diff(positive_indices_masked.astype(int))
positive_diff = np.concatenate(([0], positive_diff))
positive_diff = np.where(positive_diff > 0, positive_diff, 0)  # Keep only positive differences
print(f"Max Positive Distance: {np.max(positive_bin_centers)}, Min Positive Distance: {np.min(positive_bin_centers)}")
print(f"Max Negative Distance: {np.max(negative_bin_centers)}, Min Negative Distance: {np.min(negative_bin_centers)}")

positive_spikes = positive_bin_centers[np.where(positive_diff > 0)[0]]
negative_spikes = negative_bin_centers[np.where(negative_diff < 0)[0]]
# sort the positive spikes in ascending order
positive_spikes = np.sort(positive_spikes)

used_negative_spikes = []
rois = []
for i, positive_spike in enumerate(positive_spikes):
    larger_negative_spikes = negative_spikes[negative_spikes > positive_spike]
    right_closest_negative_index = np.argmin(larger_negative_spikes - positive_spike)
    if right_closest_negative_index < len(negative_bin_centers):
        right_closest_negative_spike = larger_negative_spikes[right_closest_negative_index]
        if right_closest_negative_spike not in used_negative_spikes:
            used_negative_spikes.append(right_closest_negative_spike)
            roi = (positive_spike, right_closest_negative_spike)
            rois.append(roi)

rois_filtered = []
roi_counts = []
for i, (start, end) in enumerate(rois):
    should_add_region = False
    line_count = 0
    for wire_dist in midpoint_dists_wrt_center: 
        if start <= wire_dist <= end:
            # Append the region to the list
            line_count += 1
            should_add_region = True
            
    if should_add_region:
        rois_filtered.append((start, end))
        roi_counts.append(line_count)

print(f"Positive Spikes: {positive_spikes}, Negative Spikes: {negative_spikes}")

plt.figure()
plt.bar(negative_bin_centers, negative_grads, width=1, color='blue', alpha=0.7)
plt.bar(positive_bin_centers, positive_grads, width=1, color='green', alpha=0.7)
plt.vlines(midpoint_dists_wrt_center, min(negative_grads), max(positive_grads), color='red', alpha=0.7, label='wire midpoints')
plt.hlines(- wire_detector.grad_bin_avg_threshold, min(negative_bin_centers), max(negative_bin_centers), color='green', alpha=0.7, label='negative threshold')
plt.hlines(wire_detector.grad_bin_avg_threshold, min(positive_bin_centers), max(positive_bin_centers), color='orange', alpha=0.7, label='positive threshold')
plt.xlabel('Distance from center (pixels)')
plt.ylabel('Gradient magnitude')
plt.title('Depth Gradient Histogram')
plt.show()

plt.figure()
plt.vlines(positive_spikes, 0, 1, color='green', alpha=0.7, label='Positive Spikes')
plt.vlines(negative_spikes, -1, 0, color='blue', alpha=0.7, label='Negative Spikes')
plt.vlines(midpoint_dists_wrt_center, -1, 1, color='red', alpha=0.7, label='wire midpoints')
for start, end in rois_filtered:
    plt.axvspan(start, end, color='yellow', alpha=0.5)
    plt.axvline(x=start, color='green', linestyle='--')
    plt.axvline(x=end, color='green', linestyle='--')

plt.xlabel('Distance from center (pixels)')
plt.ylabel('Gradient Difference')
plt.title('Gradient Difference Histogram')
plt.legend()
plt.show()