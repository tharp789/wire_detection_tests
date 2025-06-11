import numpy as np
import cv2
from scipy.stats import circmean

from wire_detection_utils import WireDetector, fold_angles_from_0_to_pi, perpendicular_angle_rad, get_length_of_center_line_across_image, peak_hist_into_wires, project_image_to_axis, find_closest_distance_from_points_to_line_3d

class WireDetectorCPU(WireDetector):
    def __init__(self, wire_detection_config, camera_intrinsics):
        super().__init__(wire_detection_config, camera_intrinsics)

    def get_hough_lines(self, rgb_image):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        seg_mask = cv2.Canny(gray, self.low_canny_threshold, self.high_canny_threshold, apertureSize=3)
        cartesian_lines = cv2.HoughLinesP(seg_mask, 1, np.pi/180, self.hough_vote_threshold, minLineLength=self.min_line_threshold, maxLineGap=10)
        if cartesian_lines is None:
            return None, None, None
    
    def get_xy_depth_gradients(self, depth_image):
        """
        Compute the x and y gradients of the depth image using Sobel filters.
        """
        depth_gradient_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=11)
        depth_gradient_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=11)
        return depth_gradient_x, depth_gradient_y

    # def find_regions_of_interest(self, depth, avg_angle, midpoint_dists_wrt_center, viz_img=None):

    #     depth_gradient_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=11)
    #     depth_gradient_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=11)
    #     perp_angle = perpendicular_angle_rad(avg_angle)
    #     depth_gradient = depth_gradient_x * np.cos(perp_angle) + depth_gradient_y * np.sin(perp_angle)

    #     distance, depth_gradient_1d = project_image_to_axis(depth_gradient, perp_angle)
    #     depth_gradient_1d = np.abs(depth_gradient_1d)
    #     depth_gradient_1d = depth_gradient_1d / np.max(depth_gradient_1d)

    #     dist_hist, bin_edges = np.histogram(distance, bins=np.arange(np.min(distance), np.max(distance), 1), weights=depth_gradient_1d)
    #     bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    #     dist_hist = dist_hist / np.max(dist_hist)

    #     threshold = self.grad_bin_avg_threshold
    #     mask = dist_hist > threshold
    #     mask_diff = np.diff(mask.astype(int))
    #     mask_diff = np.concatenate(([0], mask_diff))

    #     start_indices = np.where(mask_diff == 1)[0]
    #     end_indices = np.where(mask_diff == -1)[0]

    #     if len(start_indices) == 0 or len(end_indices) == 0:
    #         return [], []

    #     if start_indices[0] > end_indices[0]:
    #         start_indices = np.insert(start_indices, 0, 0)
    #     if len(start_indices) > len(end_indices):
    #         end_indices = np.append(end_indices, len(mask) - 1)
    #     if len(end_indices) > len(start_indices):
    #         start_indices = np.insert(start_indices, 0, 0)

    #     assert len(start_indices) == len(end_indices), "Mismatch in start and end indices length"

    #     regions_of_interest = []
    #     roi_line_count = []
    #     for start, end in zip(start_indices, end_indices):
    #         if bin_centers[start] < bin_centers[end]:
    #             start = bin_centers[start]
    #             end = bin_centers[end]
    #             should_add_region = False
    #             line_count = 0
    #             for wire_dist in midpoint_dists_wrt_center: 
    #                 if start <= wire_dist <= end:
    #                     # Append the region to the list
    #                     line_count += 1
    #                     should_add_region = True
                        
    #             if should_add_region:
    #                 regions_of_interest.append((start, end))
    #                 roi_line_count.append(line_count)

    #     # limit the count per region to the max number of wires per ROI
    #     roi_line_count = np.array(roi_line_count)
    #     roi_line_count = np.clip(roi_line_count, 0, self.max_wire_per_roi)
    #     return regions_of_interest, roi_line_count

    # def roi_to_point_clouds(self, rois, avg_angle, depth_image, viz_img=None):
    #     """
    #     Convert a region of interest (ROI) to a binary mask.

    #     Parameters:
    #         roi (np.ndarray): Array of shape (N, 4) where each row is [x1, y1, x2, y2].
    #         img_shape (tuple): Shape of the image (height, width).

    #     Returns:
    #         np.ndarray: Binary mask of the same shape as the image.
    #     """

    #     img_shape = depth_image.shape[:2]  # (height, width)
    #     img_center = np.array([img_shape[1] // 2, img_shape[0] // 2])  # (x, y)
    #     if viz_img is not None:
    #         viz_mask = np.zeros(viz_img.shape[:2], dtype=np.uint8)

    #     perp_angle = perpendicular_angle_rad(avg_angle)  # Get the perpendicular angle in radians

    #     # Example angle in radians
    #     # Make sure avg_angle is in radians — if it's in degrees, convert with np.radians()
    #     roi_depths = []
    #     roi_rgb = []
    #     for start, end in rois:
    #         center_dist = 0.5 * (start + end)  # scalar, along direction of avg_angle
    #         length = abs(end - start)          # width of the ROI

    #         # Compute center offset in image coordinates
    #         dx = center_dist * np.cos(perp_angle)
    #         dy = center_dist * np.sin(perp_angle)
    #         center_coords = (img_center[0] + dx, img_center[1] + dy)

    #         # Define rectangle size: length along projected axis, large height perpendicular
    #         size = (length, img_shape[1] * 2)  # (width, height)

    #         # Create rotated rectangle
    #         rect = (center_coords, size, np.degrees(perp_angle))
    #         box = cv2.boxPoints(rect).astype(int)

    #         # Draw box on mask
    #         if viz_img is not None:
    #             cv2.fillConvexPoly(viz_mask, box, 255)

    #         single_roi_mask = np.zeros(img_shape, dtype=np.uint8)
    #         cv2.fillConvexPoly(single_roi_mask, box, 255)
    #         single_roi_depth_image = cv2.bitwise_and(depth_image, depth_image, mask=single_roi_mask)
    #         if viz_img is not None:
    #             single_roi_rgb_image = cv2.bitwise_and(viz_img, viz_img, mask=single_roi_mask)
    #             roi_rgb.append(single_roi_rgb_image)
    #         roi_depths.append(single_roi_depth_image)

    #     if viz_img is not None:
    #         masked_viz_img = cv2.bitwise_and(viz_img, viz_img, mask=viz_mask)
    #         depth_img_masked = cv2.bitwise_and(depth_image, depth_image, mask=viz_mask)
    #         return roi_depths, depth_img_masked, roi_rgb, masked_viz_img
    #     else:
    #         return roi_depths, None, None, None
    
    # def ransac_line_fitting(self, points, avg_angle, num_lines):
    #     """
    #     RANSAC line fitting algorithm.

    #     Parameters:
    #         points (np.ndarray): Array of shape (N, 2) where each row is a point (x, y).
    #         num_iterations (int): Number of RANSAC iterations.
    #         threshold (float): Distance threshold to consider a point as an inlier.

    #     Returns:
    #         tuple: Best line parameters (slope, intercept) and inliers.
    #     """
    #     avg_angle = fold_angles_from_0_to_pi(avg_angle)
    #     best_lines = []
    #     line_inlier_counts = []
    #     for i in range(num_lines):
    #         best_inliers_mask = None
    #         best_inlier_count = 0
    #         best_line = None

    #         for _ in range(self.ransac_max_iters):
    #             # Randomly select two points
    #             sample_indices = np.random.choice(points.shape[0], 2, replace=False)
    #             p1, p2 = points[sample_indices]
                
    #             pitch_angle = np.arctan2(np.abs(p2[2] - p1[2]), np.linalg.norm(p2[:2] - p1[:2]))
    #             pitch_angle = fold_angles_from_0_to_pi(pitch_angle)
    #             if pitch_angle > self.vert_angle_maximum_rad and pitch_angle < np.pi - self.vert_angle_maximum_rad:
    #                 continue

    #             yaw_angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    #             yaw_angle = fold_angles_from_0_to_pi(yaw_angle)
                
    #             angle_diff = np.abs(yaw_angle - avg_angle)
    #             if angle_diff > np.pi / 2:
    #                 angle_diff = np.abs(np.pi - angle_diff)
    #             if angle_diff > self.horz_angle_diff_maximum_rad:
    #                 continue

    #             # Calculate line parameters
    #             distances = find_closest_distance_from_points_to_line_3d(points, np.array([p1, p2]))

    #             # Find inliers
    #             inlier_mask = distances <= self.inlier_threshold_m
    #             inlier_count = np.count_nonzero(inlier_mask)

    #             # Update best line if current one has more inliers
    #             if inlier_count > best_inlier_count:
    #                 best_inlier_count = inlier_count
    #                 best_line = (p1, p2)
    #                 best_inliers_mask = inlier_mask
            
    #         # Remove inliers from the dataset for the next iteration
    #         if best_line is None:
    #             break
    #         best_lines.append(best_line)
    #         line_inlier_counts.append(best_inlier_count)

    #         if num_lines > 1 and best_inliers_mask is not None:
    #             points = points[~best_inliers_mask]
    #             if len(points) <= 2:
    #                 break

    #     # combine lines if there z height is withing the inlier threshold
    #     if len(best_lines) > 1:
    #         combined_lines = []
    #         combined_inlier_counts = []
    #         for i, line in enumerate(best_lines):
    #             p1, p2 = line
    #             avg_height = (p1[2] + p2[2]) / 2

    #             for j, (cp1, cp2) in enumerate(combined_lines):
    #                 combined_avg_height = (cp1[2] + cp2[2]) / 2
    #                 if np.abs(combined_avg_height - avg_height) <= self.inlier_threshold_m * 2:

    #                     d1 = np.linalg.norm(p1 - cp1) + np.linalg.norm(p2 - cp2)
    #                     d2 = np.linalg.norm(p1 - cp2) + np.linalg.norm(p2 - cp1)
    #                     combined_inlier_count = combined_inlier_counts[j]
    #                     line_inlier_count = line_inlier_counts[i]
    #                     total_inlier_count = combined_inlier_count + line_inlier_count
    #                     if d1 < d2:
    #                         new_p1 = (cp1 * combined_inlier_count + p1 * line_inlier_count) / total_inlier_count
    #                         new_p2 = (cp2 * combined_inlier_count + p2 * line_inlier_count) / total_inlier_count
    #                     else:
    #                         new_p1 = (cp1 * combined_inlier_count + p2 * line_inlier_count) / total_inlier_count
    #                         new_p2 = (cp2 * combined_inlier_count + p1 * line_inlier_count) / total_inlier_count

    #                     combined_lines[j] = (new_p1, new_p2)
    #                     combined_inlier_counts[j] += line_inlier_counts[i]
    #                     break
    #             else:
    #                 combined_lines.append(line)
    #                 combined_inlier_counts.append(line_inlier_counts[i])

    #         return combined_lines, combined_inlier_counts

    #     return best_lines, line_inlier_counts
    
    # def ransac_on_rois(self, rois, roi_line_counts, avg_angle, depth_image, viz_img=None):
    #     """
    #     Find wires in 3D from the regions of interest.

    #     Parameters:
    #         roi_depths (list): List of depth images for each ROI.
    #         roi_rgbs (list): List of RGB images for each ROI.
    #         avg_angle (float): Average angle of the wires in radians.
    #         roi_line_count (list): List of line counts for each ROI.

    #     Returns:
    #         fitted_lines (list): List of fitted lines in 3D.
    #     """
    #     fitted_lines = []
    #     line_inlier_counts = []
    #     roi_pcs = []
    #     roi_point_colors = []
    #     roi_depths, depth_img_masked, roi_rgbs, masked_viz_img = self.roi_to_point_clouds(rois, avg_angle, depth_image, viz_img=viz_img)
    #     if roi_rgbs is None:
    #         roi_rgbs = [None] * len(roi_depths)
    #     for roi_depth, roi_rgb, line_count in zip(roi_depths, roi_rgbs, roi_line_counts):
    #         # convert depth image to point cloud
    #         points, colors = self.depth_to_pointcloud(roi_depth, rgb=roi_rgb, depth_clip=[self.min_depth_clip, self.max_depth_clip])
    #         roi_pcs.append(points)
    #         if colors is not None:
    #             colors = (np.array(colors) / 255.0)[:,::-1]
    #             roi_point_colors.append(colors)
    #         lines, line_inlier_count = self.ransac_line_fitting(points, avg_angle, line_count)
    #         fitted_lines += lines
    #         line_inlier_counts += line_inlier_count

    #     return fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors if roi_point_colors else None, masked_viz_img if viz_img is not None else None
    
    # def detect_3d_wires(self, rgb_image, depth_image, generate_viz = False):
    #     """
    #     Find wires in 3D from the RGB and depth images.
    #     """
    #     wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = self.detect_wires_2d(rgb_image)

    #     regions_of_interest, roi_line_counts = self.find_regions_of_interest(depth_image, avg_angle, midpoint_dists_wrt_center)

    #     if generate_viz:
    #         fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = self.ransac_on_rois(regions_of_interest, roi_line_counts, avg_angle, depth_image, viz_img=rgb_image)
    #     else:
    #         fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = self.ransac_on_rois(regions_of_interest, roi_line_counts, avg_angle, depth_image)

    #     return fitted_lines, rgb_masked
    
    # def depth_to_pointcloud(self, depth_image, rgb=None, depth_clip=[0.5, 10.0]):
    #     """
    #     Convert a depth image to a 3D point cloud.

    #     Parameters:
    #         depth_image (np.ndarray): 2D depth image.
    #         camera_intrinsics (np.ndarray): Camera intrinsic matrix.

    #     Returns:
    #         np.ndarray: 3D point cloud of shape (N, 3).
    #     """
    #     H, W = depth_image.shape

    #     # Create meshgrid of pixel coordinates
    #     x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))  # shape: (H, W)

    #     # Compute 3D points
    #     flatted_coord = np.column_stack((x_coords.flatten(), y_coords.flatten(), np.ones_like(x_coords.flatten())))
    #     z_coords = depth_image.flatten()
    #     valid_mask = ~np.isnan(z_coords) & (z_coords > depth_clip[0]) & (z_coords < depth_clip[1])
    #     inv_camera_intrinsics = np.linalg.inv(self.camera_intrinsics)

    #     rays = np.dot(inv_camera_intrinsics, flatted_coord.T).T
    #     points = rays * z_coords.reshape(-1, 1)
    #     points = points[valid_mask]
    #     if rgb is not None:
    #         rgb = rgb.reshape(-1, 3)
    #         rgb = rgb[valid_mask]
    #     return points, rgb if rgb is not None else None
    