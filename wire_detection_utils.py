import numpy as np
import cv2
from scipy.stats import circmean
from scipy.signal import find_peaks

class WireDetector:
    def __init__(self, line_threshold, low_canny_threshold, high_canny_threshold, pixel_binning_size, bin_avg_threshold_multiplier):
        self.line_threshold = line_threshold
        self.low_canny_threshold = low_canny_threshold
        self.high_canny_threshold = high_canny_threshold
        self.pixel_binning_size = pixel_binning_size
        self.bin_avg_threshold_multiplier = bin_avg_threshold_multiplier
        
        self.img_height = None
        self.img_width = None
        self.img_shape = None
        self.cx = None
        self.cy = None
        self.line_length = None

    def get_hough_lines(self, seg_mask):
        seg_coords = np.argwhere(seg_mask==255)
        seg_coords = seg_coords[:, [1, 0]]

        cartesian_lines = cv2.HoughLinesP(seg_mask, 1, np.pi/180, self.line_threshold)
        if cartesian_lines is None:
            return None, None, None, None, None
        
        cartesian_lines = np.squeeze(cartesian_lines,axis=1)
        line_lengths = np.linalg.norm(cartesian_lines[:, 2:4] - cartesian_lines[:, 0:2], axis=1).astype(int)
        cartesian_lines = cartesian_lines[line_lengths > 10]
        line_lengths = line_lengths[line_lengths > 10]

        if len(cartesian_lines) == 0:
            return None, None, None, None, None

        line_angles = np.arctan2(
            cartesian_lines[:, 3] - cartesian_lines[:, 1],  # y2 - y1
            cartesian_lines[:, 2] - cartesian_lines[:, 0]   # x2 - x1
        )
        avg_angle = circmean(line_angles, high=np.pi, low=-np.pi)

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
        return cartesian_lines, line_lengths, center_line, avg_angle, seg_coords

    def get_line_instance_locations(self, cartesian_lines, line_lengths, center_line, avg_angle, seg_coords):
        image_perp_distance = get_length_of_center_line_across_image(self.img_height, self.img_width, perpendicular_angle_rad(avg_angle))
        bins = np.arange(- image_perp_distance // 2, image_perp_distance // 2 + self.pixel_binning_size, self.pixel_binning_size)

        pixel_dists_wrt_center = self.compute_perpendicular_distance(center_line, cartesian_lines)
        hist, bin_edges = np.histogram(pixel_dists_wrt_center, bins=bins)
        
        # find a threshold for where to count wire peaks based on count
        bin_threshold = self.bin_avg_threshold_multiplier * np.mean(hist[hist > 0])

        wire_distances_wrt_center = peak_hist_into_wires(hist, bin_edges, pixel_dists_wrt_center, bin_threshold)

        sin_offset, cos_offset = np.sin(avg_angle + np.pi / 2), np.cos(avg_angle + np.pi / 2)
        new_midpoints = np.column_stack((
            self.cx + wire_distances_wrt_center * cos_offset,
            self.cy + wire_distances_wrt_center * sin_offset
        ))

        dists = np.linalg.norm(seg_coords[:, None] - new_midpoints, axis=2)
        closest_indices = np.argmin(dists, axis=0)
        wire_midpoints = seg_coords[closest_indices]

        # Compute wire lines in a vectorized manner
        new_x0 = wire_midpoints[:, 0] + self.line_length * np.cos(avg_angle)
        new_y0 = wire_midpoints[:, 1] + self.line_length * np.sin(avg_angle)
        new_x1 = wire_midpoints[:, 0] - self.line_length * np.cos(avg_angle)
        new_y1 = wire_midpoints[:, 1] - self.line_length * np.sin(avg_angle)

        wire_lines = np.column_stack((new_x0, new_y0, new_x1, new_y1)).astype(int)

        return wire_lines, wire_midpoints, hist, bin_edges, bin_threshold, wire_distances_wrt_center

    def detect_wires_2d(self, seg_mask):
        cartesian_lines, line_lengths, center_line, avg_angle, seg_coords = self.get_hough_lines(seg_mask)
        if cartesian_lines is not None: 
            wire_lines, wire_midpoints, _ , _ , _ , midpoint_dists_wrt_center = self.get_line_instance_locations(cartesian_lines, line_lengths, center_line, avg_angle, seg_coords)
            wire_lines = np.array(wire_lines)
            wire_midpoints = np.array(wire_midpoints)
        else:
            wire_lines = np.array([])
            wire_midpoints = np.array([])
        return wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center
        
    def create_seg_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        seg_mask = cv2.Canny(gray, self.low_canny_threshold, self.high_canny_threshold, apertureSize=3)
        return seg_mask
    
    def get_pixels_from_lines(self, lines):
        img_mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        for x1, y1, x2, y2 in lines:
            cv2.line(img_mask, (x1, y1), (x2, y2), 255, 1)
        pixels = np.argwhere(img_mask == 255)
        pixels = pixels[:, [1, 0]]  # Convert to (x, y) format
        return pixels
    
    def compute_perpendicular_distance(self, center_line, lines):
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

def get_length_of_center_line_across_image(image_height, image_width, angle):
    assert isinstance(image_height, int) and isinstance(image_width, int), "Image dimensions must be integers"
    assert isinstance(angle, (int, float)), "Angle must be a scalar"

    # Calculate the length of the center line across the image
    angle = angle % (2 * np.pi)  # Normalize angle to [0, 2π)
    cos_angle = np.abs(np.cos(angle))
    sin_angle = np.abs(np.sin(angle))

    length = np.sqrt((image_height * sin_angle) ** 2 + (image_width * cos_angle) ** 2)
    return length

def perpendicular_angle_rad(angle_rad):
    return (angle_rad + np.pi / 2) % (2 * np.pi)
    
def clamp_angles_pi(angles):
    angles = np.asarray(angles)  # Ensure input is an array
    angles = angles % (2 * np.pi)  # Wrap angles into [0, 2π)
    
    angles = np.where(angles < 0, angles + np.pi, angles)  # Adjust negatives
    angles = np.where(angles > np.pi, angles - np.pi, angles)  # Adjust >π
    
    return angles.item() if np.isscalar(angles) else angles

def create_depth_viz(depth):
    min_depth = 0.5
    max_depth = 20.0
    depth[np.isnan(depth)] = min_depth
    depth[np.isinf(depth)] = min_depth
    depth[np.isneginf(depth)] = min_depth

    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = (depth * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth

def compute_yaw_from_3D_points(points):
    '''
    Compute the yaw from a set of 3D points.
    Assumes z is up and y is forward, x is right.
    '''

    mean = np.mean(points, axis=0)
    centered_points = points - mean
    _, _, Vt = np.linalg.svd(centered_points)
    
    # Principal direction (first principal component)
    direction = Vt[0]  # First row of Vt is the principal direction
    
    # Extract x and y components
    dx, dy = direction[0], direction[1]

    # Compute yaw angle
    yaw_rad = np.arctan2(dy, dx)
    
    return yaw_rad

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

def roi_to_mask(rois, perp_angle, depth_image, viz_img=None):
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
    mask = np.zeros(img_shape, dtype=np.uint8)

    # Example angle in radians
    # Make sure avg_angle is in radians — if it's in degrees, convert with np.radians()
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
        cv2.fillConvexPoly(mask, box, 255)

    if viz_img is not None:
        masked_viz_img = cv2.bitwise_and(viz_img, viz_img, mask=mask)
    
    depth_masked = cv2.bitwise_and(depth_image, depth_image, mask=mask)
    return depth_masked, masked_viz_img if viz_img is not None else None
    
 