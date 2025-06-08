import numpy as np
import cv2
from scipy.stats import circmean
from scipy.signal import find_peaks

class WireDetector:
    def __init__(self, wire_detection_config):
        self.hough_vote_threshold = wire_detection_config['hough_vote_threshold']
        self.min_line_threshold = wire_detection_config['min_line_threshold']
        self.pixel_binning_size = wire_detection_config['pixel_binning_size']
        self.low_canny_threshold = wire_detection_config['low_canny_threshold']
        self.high_canny_threshold = wire_detection_config['high_canny_threshold']
        self.line_bin_avg_threshold_multiplier = wire_detection_config['line_bin_avg_threshold_multiplier']

        self.grad_bin_avg_threshold_multiplier = wire_detection_config['grad_bin_avg_threshold_multiplier']
        
        self.ransac_max_iters = wire_detection_config['ransac_max_iters']
        self.inlier_threshold_m = wire_detection_config['inlier_threshold_m']
        self.vert_angle_maximum_rad = wire_detection_config['vert_angle_maximum_rad']
        self.horz_angle_diff_maximum_rad = wire_detection_config['horz_angle_diff_maximum_rad']

        self.img_height = None
        self.img_width = None
        self.img_shape = None
        self.cx = None
        self.cy = None
        self.line_length = None

    def get_hough_lines(self, seg_mask):
        seg_coords = np.argwhere(seg_mask==255)
        seg_coords = seg_coords[:, [1, 0]]

        cartesian_lines = cv2.HoughLinesP(seg_mask, 1, np.pi/180, self.hough_vote_threshold, minLineLength=self.min_line_threshold, maxLineGap=10)
        if cartesian_lines is None:
            return None, None, None

        cartesian_lines = np.squeeze(cartesian_lines,axis=1)

        if len(cartesian_lines) == 0:
            return None, None, None

        line_angles = np.arctan2(
            cartesian_lines[:, 3] - cartesian_lines[:, 1],  # y2 - y1
            cartesian_lines[:, 2] - cartesian_lines[:, 0]   # x2 - x1
        )
        avg_angles = fold_angles_from_0_to_pi(line_angles)
        avg_angle = circmean(avg_angles, high=np.pi, low=-np.pi)

        if self.img_shape == None:
            self.img_shape = seg_mask.shape
            self.img_height, self.img_width = self.img_shape
            self.cx, self.cy = self.img_shape[1] // 2, self.img_shape[0] // 2
            self.line_length = max(self.img_shape[1], self.img_shape[0]) * 2

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

        wire_distances_wrt_center = peak_hist_into_wires(hist, bin_edges, pixel_dists_wrt_center, bin_threshold)

        sin_offset, cos_offset = np.sin(avg_angle + np.pi / 2), np.cos(avg_angle + np.pi / 2)
        wire_midpoints = np.column_stack((
            self.cx + wire_distances_wrt_center * cos_offset,
            self.cy + wire_distances_wrt_center * sin_offset
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

        depth_gradient_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=11)
        depth_gradient_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=11)
        perp_angle = perpendicular_angle_rad(avg_angle)
        depth_gradient = depth_gradient_x * np.cos(perp_angle) + depth_gradient_y * np.sin(perp_angle)

        distance, depth_gradient_1d = project_image_to_axis(depth_gradient, perp_angle)
        depth_gradient_1d = np.abs(depth_gradient_1d)
        depth_gradient_1d = depth_gradient_1d / np.max(depth_gradient_1d)

        dist_hist, bin_edges = np.histogram(distance, bins=np.arange(np.min(distance), np.max(distance), 1), weights=depth_gradient_1d)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        dist_hist = dist_hist / np.max(dist_hist)

        threshold = self.grad_bin_avg_threshold_multiplier * np.mean(dist_hist)
        mask = dist_hist > threshold
        mask_diff = np.diff(mask.astype(int))
        mask_diff = np.concatenate(([0], mask_diff))

        start_indices = np.where(mask_diff == 1)[0]
        end_indices = np.where(mask_diff == -1)[0]
        # if ensure that the first start index is not after the first end index
        if start_indices[0] > end_indices[0]:
            start_indices = np.insert(start_indices, 0, 0)
        if len(start_indices) > len(end_indices):
            end_indices = np.append(end_indices, len(mask) - 1)
        if len(end_indices) > len(start_indices):
            start_indices = np.append(0, start_indices)
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
    
    def ransac_on_rois(self, rois, roi_line_counts, avg_angle, depth_image, camera_intrinsics, viz_img=None):
        """
        Find wires in 3D from the regions of interest.

        Parameters:
            roi_depths (list): List of depth images for each ROI.
            roi_rgbs (list): List of RGB images for each ROI.
            camera_intrinsics (np.ndarray): Camera intrinsic matrix.
            avg_angle (float): Average angle of the wires in radians.
            roi_line_count (list): List of line counts for each ROI.

        Returns:
            fitted_lines (list): List of fitted lines in 3D.
        """
        fitted_lines = []
        roi_pcs = []
        roi_point_colors = []
        roi_depths, depth_img_masked, roi_rgbs, masked_viz_img = self.roi_to_point_clouds(rois, avg_angle, depth_image, viz_img=viz_img)

        for roi_depth, roi_rgb, line_count in zip(roi_depths, roi_rgbs, roi_line_counts):
            # convert depth image to point cloud
            points, colors = depth_to_pointcloud(roi_depth, camera_intrinsics, roi_rgb, depth_clip=[0.5, 15.0])
            roi_pcs.append(points)
            if colors is not None:
                colors = (np.array(colors) / 255.0)[:,::-1]
                roi_point_colors.append(colors)
            lines = ransac_line_fitting(points, avg_angle, num_lines=line_count, num_iterations=self.ransac_max_iters, inlier_threshold=self.inlier_threshold_m, vert_angle_thresh=self.vert_angle_maximum_rad, horiz_angle_thresh=self.horz_angle_diff_maximum_rad)
            fitted_lines += lines

        assert len(fitted_lines) == sum(roi_line_counts), f"Mismatch in fitted lines count: {len(fitted_lines)} vs {sum(roi_line_counts)}"

        return fitted_lines, roi_pcs, roi_point_colors if roi_point_colors else None
    
    def find_3d_wires(self, rgb_image, depth_image, camera_intrinsics, viz_img=None):
        """
        Find wires in 3D from the RGB and depth images.

        Parameters:
            rgb_image (np.ndarray): RGB image.
            depth_image (np.ndarray): Depth image.
            camera_intrinsics (np.ndarray): Camera intrinsic matrix.

        Returns:
            fitted_lines (list): List of fitted lines in 3D.
            roi_pcs (list): List of point clouds for each ROI.
            roi_point_colors (list): List of colors for each point in the ROIs.
        """
        wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = self.detect_wires_2d(rgb_image)
        if len(wire_lines) == 0:
            return [], [], None

        rois, roi_line_counts = self.find_regions_of_interest(depth_image, avg_angle, midpoint_dists_wrt_center)
        fitted_lines, roi_pcs, roi_point_colors = self.ransac_on_rois(rois, roi_line_counts, avg_angle, depth_image, camera_intrinsics, viz_img=viz_img)

        return fitted_lines, roi_pcs, roi_point_colors


def bin_hist_into_wires(hist, bin_edges, distances_wrt_center, bin_threshold):
    """
    Computes average distances within histogram bins whose counts exceed a threshold.

    Parameters:
        hist (np.ndarray): Histogram counts.
        bin_edges (np.ndarray): Histogram bin edges (length should be len(hist)+1).
        distances_wrt_center (np.ndarray): Original distances.
        bin_threshold (float): Minimum bin count to consider.

    Returns:
        np.ndarray: Filtered average distances for valid bins.
    """
    wire_distances_wrt_center = []

    for i, counts in enumerate(hist):
        if counts >= bin_threshold:
            # Find distances within the current bin range
            binned_wire_distances = distances_wrt_center[
                (distances_wrt_center >= bin_edges[i]) & 
                (distances_wrt_center < bin_edges[i + 1])
            ]
            if binned_wire_distances.size > 0:
                avg_distance = np.mean(binned_wire_distances)
                if not np.isnan(avg_distance):
                    wire_distances_wrt_center.append(avg_distance)

    return np.array(wire_distances_wrt_center)

def peak_hist_into_wires(hist, bin_edges, distances_wrt_center, bin_threshold):
    """
    Computes average distances within histogram peaks whose counts exceed a threshold.

    Parameters:
        hist (np.ndarray): Histogram counts.
        bin_edges (np.ndarray): Histogram bin edges (length should be len(hist)+1).
        distances_wrt_center (np.ndarray): Original distances.
        bin_threshold (float): Minimum bin count to consider.

    Returns:
        np.ndarray: Filtered average distances for valid peaks.
    """
    # Find peaks in the histogram
    peaks, _ = find_peaks(hist, height=bin_threshold)

    # Find the corresponding bin center for each peak
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    distances = bin_centers[peaks]

    return np.array(distances)
    
def find_closest_point_on_3d_line(line_midpoint, yaw, target_point):
    assert line_midpoint.shape == (3,), f"Line midpoint must be a 3D point, got {line_midpoint.shape}"
    assert target_point.shape == (3,), f"Target point must be a 3D point, got {target_point.shape}"
    assert np.isscalar(yaw), f"Yaw must be a scalar, got {yaw}"

    x0, y0, z0 = line_midpoint
    xt, yt, zt = target_point

    diff_vector = np.array([xt - x0, yt - y0, zt - z0]).flatten()
    assert diff_vector.shape == (3,), f"Invalid shape for difference vector: {diff_vector.shape}"

    # Direction vector based on yaw
    direction = np.array([np.cos(yaw), np.sin(yaw), 0.0]).flatten()
    t = np.dot(diff_vector, direction) / np.dot(direction, direction)
    closest_point = np.array([x0, y0, z0]) + t * direction
    return closest_point

def find_closest_distance_from_points_to_line_3d(points, line_ends):
    assert points.shape[1] == 3, f"Points must be 3D, got shape {points.shape}"
    assert line_ends.shape[1] == 3 and line_ends.shape[0] == 2, f"Line ends must be 3D points, got shape {line_ends.shape}"

    p1, p2 = line_ends
    line_vector = p2 - p1
    line_length_squared = np.dot(line_vector, line_vector)

    # Vector from p1 to each point
    p1_to_points = points - p1
    t = np.dot(p1_to_points, line_vector) / line_length_squared

    # Clamp t to the range [0, 1]
    t_clamped = np.clip(t, 0, 1)

    # Find the closest point on the line segment
    closest_points = p1 + t_clamped[:, np.newaxis] * line_vector

    # Calculate distances from points to the closest points on the line segment
    distances = np.linalg.norm(points - closest_points, axis=1)
    
    return distances

def get_length_of_center_line_across_image(image_height, image_width, angle):
    assert isinstance(image_height, int) and isinstance(image_width, int), "Image dimensions must be integers"
    assert isinstance(angle, (int, float)), "Angle must be a scalar"

    # Calculate the length of the center line across the image
    angle = fold_angles_from_0_to_pi(angle)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    length = np.sqrt((image_height * sin_angle) ** 2 + (image_width * cos_angle) ** 2)
    return length

def perpendicular_angle_rad(angle_rad):
    return fold_angles_from_0_to_pi(angle_rad + np.pi / 2)
    
def fold_angles_from_0_to_pi(angles):
    '''
    Fold angles to the range [0, π].
    '''
    angles = np.asarray(angles)  # Ensure input is an array
    angles = angles % (2 * np.pi)  # Wrap into [0, 2π)

    # Fold anything > π into [0, π]
    folded = np.where(angles > np.pi, angles - np.pi, angles)

    return folded.item() if np.isscalar(angles) else folded

def project_image_to_axis(value_img, yaw_rad):
    """
    Projects pixels from a depth image onto an axis through the image center at a given yaw angle,
    and pairs each projection with its corresponding depth value.

    Parameters:
        depth_image (np.ndarray): 2D depth image.
        yaw_degrees (float): Yaw angle in degrees.

    Returns:
        np.ndarray: Array of shape (N, 2) where each row is (normalized_projection, depth_value)
    """
    H, W = value_img.shape
    center = np.array([W / 2.0, H / 2.0])

    # Convert yaw to radians
    axis = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])

    # Create meshgrid of pixel coordinates
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))  # shape: (H, W)
    coords = np.stack([x_coords - center[0], y_coords - center[1]], axis=2)  # shape: (H, W, 2)

    # Compute projections
    projections = (coords @ axis).flatten()  # shape: (H, W)

    values = value_img.flatten()
    return projections, values

def depth_to_pointcloud(depth_image, camera_intrinsics, rgb=None, depth_clip=[0.5, 10.0]):
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
    inv_camera_intrinsics = np.linalg.inv(camera_intrinsics)

    rays = np.dot(inv_camera_intrinsics, flatted_coord.T).T
    points = rays * z_coords.reshape(-1, 1)
    valid_mask = ~np.isnan(z_coords) & (z_coords >= depth_clip[0]) & (z_coords <= depth_clip[1])
    points = points[valid_mask]
    if rgb is not None:
        rgb = rgb.reshape(-1, 3)
        rgb = rgb[valid_mask]
    return points, rgb if rgb is not None else None

def ransac_line_fitting(points, avg_angle, num_lines = 1, num_iterations=1000, inlier_threshold=0.05, vert_angle_thresh=np.pi/15, horiz_angle_thresh=np.pi/10):
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
        best_inliers = []
        best_outliers = []
        best_line = None
        iters = 0
        while iters < num_iterations:
            # Randomly select two points
            sample_indices = np.random.choice(points.shape[0], 2, replace=False)
            p1, p2 = points[sample_indices]
            
            pitch_angle = np.arctan2(np.abs(p2[2] - p1[2]), np.linalg.norm(p2[:2] - p1[:2]))
            pitch_angle = fold_angles_from_0_to_pi(pitch_angle)
            if pitch_angle > vert_angle_thresh and pitch_angle < np.pi - vert_angle_thresh:
                iters += 1
                continue

            yaw_angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            yaw_angle = fold_angles_from_0_to_pi(yaw_angle)
            
            angle_diff = np.abs(yaw_angle - avg_angle)
            if angle_diff > np.pi / 2:
                angle_diff = np.abs(np.pi - angle_diff)
            if angle_diff > horiz_angle_thresh:
                iters += 1
                continue

            # Calculate line parameters
            distances = find_closest_distance_from_points_to_line_3d(points, np.array([p1, p2]))

            # Find inliers
            inliers = points[distances <= inlier_threshold]
            # Update best line if current one has more inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_line = (p1, p2)
                best_outliers = points[distances > inlier_threshold]
            iters += 1
        
        # Remove inliers from the dataset for the next iteration
        if best_line is not None:
            best_lines.append(best_line)
            if num_lines > 1:
                # remove inliers from points
                points = best_outliers
    return best_lines
    
 