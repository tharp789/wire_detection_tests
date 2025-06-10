import numpy as np
import cv2
from scipy.stats import circmean
from scipy.signal import find_peaks

from wire_detection_utils import fold_angles_from_0_to_pi, perpendicular_angle_rad, project_image_to_axis, peak_hist_into_wires, find_closest_distance_from_points_to_line_3d, get_length_of_center_line_across_image

class WireDetectorGPU:
    def __init__(self, wire_detection_config, camera_intrinsics):

        self.hough_vote_threshold = wire_detection_config['hough_vote_threshold']
        self.min_line_threshold = wire_detection_config['min_line_threshold']
        self.pixel_binning_size = wire_detection_config['pixel_binning_size']
        self.low_canny_threshold = wire_detection_config['low_canny_threshold']
        self.high_canny_threshold = wire_detection_config['high_canny_threshold']
        self.line_bin_avg_threshold_multiplier = wire_detection_config['line_bin_avg_threshold_multiplier']

        self.grad_bin_avg_threshold = wire_detection_config['grad_bin_avg_threshold']

        self.min_depth_clip = wire_detection_config['min_depth_clip_m']
        self.max_depth_clip = wire_detection_config['max_depth_clip_m']
        self.ransac_max_iters = wire_detection_config['ransac_max_iters']
        self.inlier_threshold_m = wire_detection_config['inlier_threshold_m']
        self.vert_angle_maximum_rad = wire_detection_config['vert_angle_maximum_rad']
        self.horz_angle_diff_maximum_rad = wire_detection_config['horz_angle_diff_maximum_rad']

        self.camera_intrinsics = camera_intrinsics

        self.img_height = None
        self.img_width = None
        self.img_shape = None
        self.cx = None
        self.cy = None
        self.line_length = None

        # Persistent GPU Mats
        self.gpu_rgb = cv2.cuda_GpuMat()
        self.gpu_depth = cv2.cuda_GpuMat()
        self.gpu_gray = cv2.cuda_GpuMat()

        # Persistent CUDA filters
        self.canny_detector = cv2.cuda.createCannyEdgeDetector(
            self.low_canny_threshold, self.high_canny_threshold, apertureSize=3
        )
        self.hough_detector = cv2.cuda.createHoughSegmentDetector(
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_vote_threshold,
            minLineLength=self.min_line_threshold,
            maxLineGap=10
        )

        self.sobel_x = cv2.cuda.createSobelFilter(cv2.CV_32F, cv2.CV_32F, 1, 0, ksize=11)
        self.sobel_y = cv2.cuda.createSobelFilter(cv2.CV_32F, cv2.CV_32F, 0, 1, ksize=11)

    def get_hough_lines(self, rgb_image):
        self.gpu_rgb.upload(rgb_image)
        self.gpu_gray = cv2.cuda.cvtColor(self.gpu_rgb, cv2.COLOR_RGB2GRAY)
        gpu_edges = self.canny_detector.detect(self.gpu_gray)
        gpu_lines = self.hough_detector.detect(gpu_edges)

        if gpu_lines is None or gpu_lines.empty():
            return None, None, None

        cartesian_lines = gpu_lines.download()
        cartesian_lines = np.squeeze(cartesian_lines, axis=1)
        if len(cartesian_lines) == 0:
            return None, None, None

        line_angles = np.arctan2(
            cartesian_lines[:, 3] - cartesian_lines[:, 1],
            cartesian_lines[:, 2] - cartesian_lines[:, 0]
        )
        avg_angles = fold_angles_from_0_to_pi(line_angles)
        avg_angle = circmean(avg_angles, high=np.pi, low=-np.pi)

        if self.img_shape is None:
            self.img_shape = rgb_image.shape[:2]
            self.img_height, self.img_width = self.img_shape
            self.cx, self.cy = self.img_width // 2, self.img_height // 2
            self.line_length = max(self.img_width, self.img_height) * 2

        cos_avg, sin_avg = np.cos(avg_angle), np.sin(avg_angle)
        x0_avg = int(self.cx + self.line_length * cos_avg)
        y0_avg = int(self.cy + self.line_length * sin_avg)
        x1_avg = int(self.cx - self.line_length * cos_avg)
        y1_avg = int(self.cy - self.line_length * sin_avg)
        center_line = np.array([x0_avg, y0_avg, x1_avg, y1_avg])
        return cartesian_lines, center_line, avg_angle

    def get_line_instance_locations(self, cartesian_lines, center_line, avg_angle):
        perp_angle = perpendicular_angle_rad(avg_angle)
        image_perp_distance = get_length_of_center_line_across_image(self.img_height, self.img_width, perp_angle)
        bins = np.arange(- image_perp_distance // 2, image_perp_distance // 2 + self.pixel_binning_size, self.pixel_binning_size)

        pixel_dists_wrt_center = self.compute_perpendicular_distance(center_line, cartesian_lines)
        hist, bin_edges = np.histogram(pixel_dists_wrt_center, bins=bins)
        
        # find a threshold for where to count wire peaks based on count
        bin_threshold = self.line_bin_avg_threshold_multiplier * np.mean(hist[hist > 0])

        wire_distances_wrt_center = peak_hist_into_wires(hist, bin_edges, pixel_dists_wrt_center, bin_threshold) * -1 # flipping the sign to match the direction of the center line

        sin_offset, cos_offset = np.sin(avg_angle + np.pi / 2), np.cos(avg_angle + np.pi / 2)
        wire_midpoints = np.column_stack((
            self.cx - wire_distances_wrt_center * cos_offset,
            self.cy - wire_distances_wrt_center * sin_offset
        ))

        # dists = np.linalg.norm(seg_coords[:, None] - new_midpoints, axis=2)
        # closest_indices = np.argmin(dists, axis=0)
        # wire_midpoints = seg_coords[closest_indices]

        # Compute wire lines in a vectorized manner
        new_x0 = wire_midpoints[:, 0] + self.line_length * np.cos(avg_angle)
        new_y0 = wire_midpoints[:, 1] + self.line_length * np.sin(avg_angle)
        new_x1 = wire_midpoints[:, 0] - self.line_length * np.cos(avg_angle)
        new_y1 = wire_midpoints[:, 1] - self.line_length * np.sin(avg_angle)

        wire_lines = np.column_stack((new_x0, new_y0, new_x1, new_y1)).astype(int)

        return wire_lines, wire_midpoints, hist, bin_edges, bin_threshold, wire_distances_wrt_center
    
    def create_seg_mask(self, rgb_image):
        """
        Create a binary segmentation mask from the RGB image using Canny edge detection.
        """
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        seg_mask = cv2.Canny(gray, self.low_canny_threshold, self.high_canny_threshold, apertureSize=3)
        return seg_mask

    def detect_wires_2d(self, rgb_image):
        seg_mask = self.create_seg_mask(rgb_image)
        cartesian_lines, center_line, avg_angle = self.get_hough_lines(seg_mask)
        if cartesian_lines is not None: 
            wire_lines, wire_midpoints, _ , _ , _ , midpoint_dists_wrt_center = self.get_line_instance_locations(cartesian_lines, center_line, avg_angle)
            wire_lines = np.array(wire_lines)
            wire_midpoints = np.array(wire_midpoints)
        else:
            wire_lines = np.array([])
            wire_midpoints = np.array([])
            midpoint_dists_wrt_center = np.array([])
            avg_angle = None
        return wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center
    
    def get_pixels_from_lines(self, lines):
        img_mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        for x1, y1, x2, y2 in lines:
            cv2.line(img_mask, (x1, y1), (x2, y2), 255, 1)
        pixels = np.argwhere(img_mask == 255)
        pixels = pixels[:, [1, 0]]  # Convert to (x, y) format
        return pixels
    
    def compute_perpendicular_distance(self, center_line, lines):
        """
        Computes the perpendicular distance from each line pixel to the center line.
        Parameters:
            center_line (np.ndarray): Coordinates of the center line in the format [x1, y1, x2, y2].
            lines (np.ndarray): Array of lines in the format [x1, y1, x2, y2].
        Returns:
            np.ndarray: Perpendicular distances from each line pixel to the center line.
        """
        x1, y1, x2, y2 = center_line
        
        # Compute coefficients A, B, C of the line equation Ax + By + C = 0
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        # Extract pixel coordinates
        pixels = self.get_pixels_from_lines(lines)
        x0 = pixels[:, 0]
        y0 = pixels[:, 1]
        
        # Compute perpendicular distances
        distances = (A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
        return distances

    def find_regions_of_interest(self, depth, avg_angle, midpoint_dists_wrt_center):
        self.gpu_depth.upload(depth.astype(np.float32))
        gpu_grad_x = self.sobel_x.apply(self.gpu_depth)
        gpu_grad_y = self.sobel_y.apply(self.gpu_depth)
        grad_x = gpu_grad_x.download()
        grad_y = gpu_grad_y.download()

        perp_angle = perpendicular_angle_rad(avg_angle)
        depth_gradient = grad_x * np.cos(perp_angle) + grad_y * np.sin(perp_angle)

        distance, depth_gradient_1d = project_image_to_axis(depth_gradient, perp_angle)
        depth_gradient_1d = np.abs(depth_gradient_1d)
        depth_gradient_1d = depth_gradient_1d / np.max(depth_gradient_1d)

        dist_hist, bin_edges = np.histogram(distance, bins=np.arange(np.min(distance), np.max(distance), 1), weights=depth_gradient_1d)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        dist_hist = dist_hist / np.max(dist_hist)

        threshold = self.grad_bin_avg_threshold
        mask = dist_hist > threshold
        mask_diff = np.diff(mask.astype(int))
        mask_diff = np.concatenate(([0], mask_diff))

        start_indices = np.where(mask_diff == 1)[0]
        end_indices = np.where(mask_diff == -1)[0]

        if len(start_indices) == 0 or len(end_indices) == 0:
            return [], []

        if start_indices[0] > end_indices[0]:
            start_indices = np.insert(start_indices, 0, 0)
        if len(start_indices) > len(end_indices):
            end_indices = np.append(end_indices, len(mask) - 1)
        if len(end_indices) > len(start_indices):
            start_indices = np.insert(start_indices, 0, 0)

        assert len(start_indices) == len(end_indices), "Mismatch in start and end indices length"

        regions_of_interest = []
        roi_line_count = []
        for start, end in zip(start_indices, end_indices):
            if bin_centers[start] < bin_centers[end]:
                start = bin_centers[start]
                end = bin_centers[end]
                should_add_region = False
                line_count = 0
                for wire_dist in midpoint_dists_wrt_center: 
                    if start <= wire_dist <= end:
                        # Append the region to the list
                        line_count += 1
                        should_add_region = True
                        
                if should_add_region:
                    regions_of_interest.append((start, end))
                    roi_line_count.append(line_count)

        return regions_of_interest, roi_line_count

    def roi_to_point_clouds(self, rois, avg_angle, depth_image, viz_img=None):
        """
        Convert a region of interest (ROI) to a binary mask.

        Parameters:
            roi (np.ndarray): Array of shape (N, 4) where each row is [x1, y1, x2, y2].
            img_shape (tuple): Shape of the image (height, width).

        Returns:
            np.ndarray: Binary mask of the same shape as the image.
        """

        img_shape = depth_image.shape[:2]  # (height, width)
        img_center = np.array([img_shape[1] // 2, img_shape[0] // 2])  # (x, y)
        if viz_img is not None:
            viz_mask = np.zeros(viz_img.shape[:2], dtype=np.uint8)

        perp_angle = perpendicular_angle_rad(avg_angle)  # Get the perpendicular angle in radians

        # Example angle in radians
        # Make sure avg_angle is in radians — if it's in degrees, convert with np.radians()
        roi_depths = []
        roi_rgb = []
        for start, end in rois:
            center_dist = 0.5 * (start + end)  # scalar, along direction of avg_angle
            length = abs(end - start)          # width of the ROI

            # Compute center offset in image coordinates
            dx = center_dist * np.cos(perp_angle)
            dy = center_dist * np.sin(perp_angle)
            center_coords = (img_center[0] + dx, img_center[1] + dy)

            # Define rectangle size: length along projected axis, large height perpendicular
            size = (length, img_shape[1] * 2)  # (width, height)

            # Create rotated rectangle
            rect = (center_coords, size, np.degrees(perp_angle))
            box = cv2.boxPoints(rect).astype(int)

            # Draw box on mask
            if viz_img is not None:
                cv2.fillConvexPoly(viz_mask, box, 255)

            single_roi_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.fillConvexPoly(single_roi_mask, box, 255)
            single_roi_depth_image = cv2.bitwise_and(depth_image, depth_image, mask=single_roi_mask)
            if viz_img is not None:
                single_roi_rgb_image = cv2.bitwise_and(viz_img, viz_img, mask=single_roi_mask)
                roi_rgb.append(single_roi_rgb_image)
            roi_depths.append(single_roi_depth_image)

        if viz_img is not None:
            masked_viz_img = cv2.bitwise_and(viz_img, viz_img, mask=viz_mask)
            depth_img_masked = cv2.bitwise_and(depth_image, depth_image, mask=viz_mask)
            return roi_depths, depth_img_masked, roi_rgb, masked_viz_img
        else:
            return roi_depths, None, None, None
    
    def ransac_line_fitting(self, points, avg_angle, num_lines):
        """
        RANSAC line fitting algorithm.

        Parameters:
            points (np.ndarray): Array of shape (N, 2) where each row is a point (x, y).
            num_iterations (int): Number of RANSAC iterations.
            threshold (float): Distance threshold to consider a point as an inlier.

        Returns:
            tuple: Best line parameters (slope, intercept) and inliers.
        """
        avg_angle = fold_angles_from_0_to_pi(avg_angle)
        best_lines = []
        for i in range(num_lines):
            best_inliers_mask = None
            best_inlier_count = 0
            best_line = None

            for _ in range(self.ransac_max_iters):
                # Randomly select two points
                sample_indices = np.random.choice(points.shape[0], 2, replace=False)
                p1, p2 = points[sample_indices]
                
                pitch_angle = np.arctan2(np.abs(p2[2] - p1[2]), np.linalg.norm(p2[:2] - p1[:2]))
                pitch_angle = fold_angles_from_0_to_pi(pitch_angle)
                if pitch_angle > self.vert_angle_maximum_rad and pitch_angle < np.pi - self.vert_angle_maximum_rad:
                    continue

                yaw_angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                yaw_angle = fold_angles_from_0_to_pi(yaw_angle)
                
                angle_diff = np.abs(yaw_angle - avg_angle)
                if angle_diff > np.pi / 2:
                    angle_diff = np.abs(np.pi - angle_diff)
                if angle_diff > self.horz_angle_diff_maximum_rad:
                    continue

                # Calculate line parameters
                distances = find_closest_distance_from_points_to_line_3d(points, np.array([p1, p2]))

                # Find inliers
                inlier_mask = distances <= self.inlier_threshold_m
                inlier_count = np.count_nonzero(inlier_mask)

                # Update best line if current one has more inliers
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_line = (p1, p2)
                    best_inliers_mask = inlier_mask
            
            # Remove inliers from the dataset for the next iteration
            if best_line is None:
                break
            best_lines.append(best_line)

            if num_lines > 1 and best_inliers_mask is not None:
                points = points[~best_inliers_mask]
                if points.shape[0] < 2:
                    break

        # combine lines if there z height is withing the inlier threshold
        if len(best_lines) > 1:
            combined_lines = []
            for line in best_lines:
                p1, p2 = line
                avg_height = (p1[2] + p2[2]) / 2

                for i, (cp1, cp2) in enumerate(combined_lines):
                    combined_avg_height = (cp1[2] + cp2[2]) / 2
                    if np.abs(combined_avg_height - avg_height) <= self.inlier_threshold_m * 2:
                        d1 = np.linalg.norm(p1 - cp1) + np.linalg.norm(p2 - cp2)
                        d2 = np.linalg.norm(p1 - cp2) + np.linalg.norm(p2 - cp1)
                        if d1 < d2:
                            combined_lines[i] = ((cp1 + p1) / 2, (cp2 + p2) / 2)
                        else:
                            combined_lines[i] = ((cp1 + p2) / 2, (cp2 + p1) / 2)
                        break
                else:
                    combined_lines.append(line)
            return combined_lines

        return best_lines
    
    def ransac_on_rois(self, rois, roi_line_counts, avg_angle, depth_image, viz_img=None):
        """
        Find wires in 3D from the regions of interest.

        Parameters:
            roi_depths (list): List of depth images for each ROI.
            roi_rgbs (list): List of RGB images for each ROI.
            avg_angle (float): Average angle of the wires in radians.
            roi_line_count (list): List of line counts for each ROI.

        Returns:
            fitted_lines (list): List of fitted lines in 3D.
        """
        fitted_lines = []
        roi_pcs = []
        roi_point_colors = []
        roi_depths, depth_img_masked, roi_rgbs, masked_viz_img = self.roi_to_point_clouds(rois, avg_angle, depth_image, viz_img=viz_img)
        if roi_rgbs is None:
            roi_rgbs = [None] * len(roi_depths)
        for roi_depth, roi_rgb, line_count in zip(roi_depths, roi_rgbs, roi_line_counts):
            # convert depth image to point cloud
            points, colors = self.depth_to_pointcloud(roi_depth, rgb=roi_rgb, depth_clip=[self.min_depth_clip, self.max_depth_clip])
            roi_pcs.append(points)
            if colors is not None:
                colors = (np.array(colors) / 255.0)[:,::-1]
                roi_point_colors.append(colors)
            lines = self.ransac_line_fitting(points, avg_angle, line_count)
            fitted_lines += lines

        return fitted_lines, roi_pcs, roi_point_colors if roi_point_colors else None, masked_viz_img if viz_img is not None else None
    
    def detect_3d_wires(self, rgb_image, depth_image, generate_viz = False):
        """
        Find wires in 3D from the RGB and depth images.
        """
        wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = self.detect_wires_2d(rgb_image)

        regions_of_interest, roi_line_counts = self.find_regions_of_interest(depth_image, avg_angle, midpoint_dists_wrt_center)

        if generate_viz:
            fitted_lines, roi_pcs, roi_point_colors, rgb_masked = self.ransac_on_rois(regions_of_interest, roi_line_counts, avg_angle, depth_image, viz_img=rgb_image)
        else:
            fitted_lines, roi_pcs, roi_point_colors, rgb_masked = self.ransac_on_rois(regions_of_interest, roi_line_counts, avg_angle, depth_image)

        return fitted_lines, rgb_masked
    
    def depth_to_pointcloud(self, depth_image, rgb=None, depth_clip=[0.5, 10.0]):
        """
        Convert a depth image to a 3D point cloud.

        Parameters:
            depth_image (np.ndarray): 2D depth image.
            camera_intrinsics (np.ndarray): Camera intrinsic matrix.

        Returns:
            np.ndarray: 3D point cloud of shape (N, 3).
        """
        H, W = depth_image.shape

        # Create meshgrid of pixel coordinates
        x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))  # shape: (H, W)

        # Compute 3D points
        flatted_coord = np.column_stack((x_coords.flatten(), y_coords.flatten(), np.ones_like(x_coords.flatten())))
        z_coords = depth_image.flatten()
        inv_camera_intrinsics = np.linalg.inv(self.camera_intrinsics)

        rays = np.dot(inv_camera_intrinsics, flatted_coord.T).T
        points = rays * z_coords.reshape(-1, 1)
        valid_mask = ~np.isnan(z_coords) & (z_coords >= depth_clip[0]) & (z_coords <= depth_clip[1])
        points = points[valid_mask]
        if rgb is not None:
            rgb = rgb.reshape(-1, 3)
            rgb = rgb[valid_mask]
        return points, rgb if rgb is not None else None
    