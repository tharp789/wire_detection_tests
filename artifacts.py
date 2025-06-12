


# vecotrized ransac, turns out to be slower than the non-vectorized version
def ransac_line_fitting(points, 
                        avg_angle, 
                        num_lines, 
                        ransac_max_iters=1000,
                        vert_angle_maximum_rad=np.pi / 4,
                        horz_angle_diff_maximum_rad=np.pi / 4,
                        inlier_threshold_m=0.05):

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
        line_inlier_counts = []
        for i in range(num_lines):
            sample_indices = np.random.randint(0, points.shape[0], size=(ransac_max_iters, 2))
            mask = sample_indices[:, 0] != sample_indices[:, 1]
            point_pairs = points[sample_indices[mask]]
            p1 = point_pairs[:, 0, :]
            p2 = point_pairs[:, 1, :]

            yaw_angles = np.arctan2(p2[:,1] - p1[:,1], p2[:,0] - p1[:,0])
            yaw_angles = fold_angles_from_0_to_pi(yaw_angles)
            angle_diff = np.abs(yaw_angles - avg_angle)
            angle_diff = np.where(angle_diff > np.pi / 2, np.abs(np.pi - angle_diff), angle_diff)

            pitch_angles = np.arctan2(np.abs(p2[:,2] - p1[:,2]), np.linalg.norm(p2[:,:2] - p1[:,:2]))
            pitch_angles = fold_angles_from_0_to_pi(pitch_angles)

            valid_mask = ((pitch_angles <= vert_angle_maximum_rad) | (pitch_angles >= np.pi - vert_angle_maximum_rad)) & (angle_diff <= horz_angle_diff_maximum_rad)

            p1_valid = p1[valid_mask]
            p2_valid = p2[valid_mask]

            if len(p1_valid) == 0:
                break

            lines = np.stack((p1_valid, p2_valid), axis=1)  # shape (M, 2, 3)
            distances = find_closest_distance_from_points_to_lines_3d(points, lines)

            inlier_mask = distances <= inlier_threshold_m
            inlier_counts = np.count_nonzero(inlier_mask, axis=1)

            best_inlier_count = np.max(inlier_counts)
            best_line_index = np.argmax(inlier_counts)
            if best_inlier_count == 0:
                break
            best_line = lines[best_line_index]
            best_inliers_mask = inlier_mask[best_line_index]
            best_lines.append(best_line)
            line_inlier_counts.append(best_inlier_count)

            if num_lines > 1 and best_inliers_mask is not None:
                points = points[~best_inliers_mask]
                if len(points) <= 2:
                    break

        # combine lines if there z height is withing the inlier threshold
        if len(best_lines) > 1:
            combined_lines = []
            combined_inlier_counts = []
            for i, line in enumerate(best_lines):
                p1, p2 = line
                avg_height = (p1[2] + p2[2]) / 2

                for j, (cp1, cp2) in enumerate(combined_lines):
                    combined_avg_height = (cp1[2] + cp2[2]) / 2
                    if np.abs(combined_avg_height - avg_height) <= inlier_threshold_m * 2:

                        d1 = np.linalg.norm(p1 - cp1) + np.linalg.norm(p2 - cp2)
                        d2 = np.linalg.norm(p1 - cp2) + np.linalg.norm(p2 - cp1)
                        combined_inlier_count = combined_inlier_counts[j]
                        line_inlier_count = line_inlier_counts[i]
                        total_inlier_count = combined_inlier_count + line_inlier_count
                        if d1 < d2:
                            new_p1 = (cp1 * combined_inlier_count + p1 * line_inlier_count) / total_inlier_count
                            new_p2 = (cp2 * combined_inlier_count + p2 * line_inlier_count) / total_inlier_count
                        else:
                            new_p1 = (cp1 * combined_inlier_count + p2 * line_inlier_count) / total_inlier_count
                            new_p2 = (cp2 * combined_inlier_count + p1 * line_inlier_count) / total_inlier_count

                        combined_lines[j] = (new_p1, new_p2)
                        combined_inlier_counts[j] += line_inlier_counts[i]
                        break
                else:
                    combined_lines.append(line)
                    combined_inlier_counts.append(line_inlier_counts[i])

            return combined_lines, combined_inlier_counts

        return best_lines, line_inlier_counts




# multiprocessed version of ransac line fitting
    def ransac_on_rois_async(self, rois, roi_line_counts, avg_angle, depth_image, viz_img=None):
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
        line_inlier_counts = []
        roi_pcs = []
        roi_point_colors = []
        roi_depths, depth_img_masked, roi_rgbs, masked_viz_img = self.roi_to_point_clouds(rois, avg_angle, depth_image, viz_img=viz_img)
        if roi_rgbs is None:
            roi_rgbs = [None] * len(roi_depths)
        args_list = [
            (self.inv_camera_intrinsics, roi_depth, roi_rgb, line_count, avg_angle,
                self.min_depth_clip, self.max_depth_clip, 
                self.ransac_max_iters, self.vert_angle_maximum_rad, 
                self.horz_angle_diff_maximum_rad, self.inlier_threshold_m)
            for roi_depth, roi_rgb, line_count in zip(roi_depths, roi_rgbs, roi_line_counts)
        ]
        start_time = time.perf_counter()
        results = self.pool.map(process_roi, args_list)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        for lines, line_inlier_count, points, roi_color in results:
            fitted_lines += lines
            line_inlier_counts += line_inlier_count
            roi_pcs.append(points)
            roi_point_colors.append(roi_color)

        # Filter out None values from roi_point_colors
        roi_point_colors = [color for color in roi_point_colors if color is not None]
        if not fitted_lines:
            return [], [], [], None, masked_viz_img if viz_img is not None else None
        
        return fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors if roi_point_colors else None, masked_viz_img if viz_img is not None else None
    
    def detect_3d_wires(self, rgb_image, depth_image, generate_viz = False):
        """
        Find wires in 3D from the RGB and depth images.
        """
        wire_lines, wire_midpoints, avg_angle, midpoint_dists_wrt_center = self.detect_wires_2d(rgb_image)

        regions_of_interest, roi_line_counts = self.find_regions_of_interest(depth_image, avg_angle, midpoint_dists_wrt_center)

        if generate_viz:
            fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = self.ransac_on_rois_sync(regions_of_interest, roi_line_counts, avg_angle, depth_image, viz_img=rgb_image)
        else:
            fitted_lines, line_inlier_counts, roi_pcs, roi_point_colors, rgb_masked = self.ransac_on_rois_sync(regions_of_interest, roi_line_counts, avg_angle, depth_image)

        return fitted_lines, rgb_masked

def process_roi(args):
    inv_camera_intrinsics, roi_depth, roi_rgb, line_count, avg_angle, min_depth, max_depth, \
        ransac_max_iters, vert_angle_maximum_rad, horz_angle_diff_maximum_rad, inlier_threshold_m = args
    points, colors = depth_to_pointcloud(roi_depth, inv_camera_intrinsics, rgb=roi_rgb, depth_clip=[min_depth, max_depth])
    roi_point_colors = None
    if colors is not None:
        roi_point_colors = (np.array(colors) / 255.0)[:, ::-1]
    lines, line_inlier_count = ransac_line_fitting(points, avg_angle, line_count, 
                                                    ransac_max_iters=ransac_max_iters,
                                                    vert_angle_maximum_rad=vert_angle_maximum_rad,
                                                    horz_angle_diff_maximum_rad=horz_angle_diff_maximum_rad,
                                                    inlier_threshold_m=inlier_threshold_m)
    return lines, line_inlier_count, points, roi_point_colors

def ransac_line_fitting(points, 
                        avg_angle, 
                        num_lines, 
                        ransac_max_iters=1000,
                        vert_angle_maximum_rad=np.pi / 4,
                        horz_angle_diff_maximum_rad=np.pi / 4,
                        inlier_threshold_m=0.05):

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
        line_inlier_counts = []
        for i in range(num_lines):
            best_inliers_mask = None
            best_inlier_count = 0
            best_line = None

            for _ in range(ransac_max_iters):
                # Randomly select two points
                sample_indices = np.random.choice(points.shape[0], 2, replace=False)
                p1, p2 = points[sample_indices]
                
                pitch_angle = np.arctan2(np.abs(p2[2] - p1[2]), np.linalg.norm(p2[:2] - p1[:2]))
                pitch_angle = fold_angles_from_0_to_pi(pitch_angle)
                if pitch_angle > vert_angle_maximum_rad and pitch_angle < np.pi - vert_angle_maximum_rad:
                    continue

                yaw_angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                yaw_angle = fold_angles_from_0_to_pi(yaw_angle)
                
                angle_diff = np.abs(yaw_angle - avg_angle)
                if angle_diff > np.pi / 2:
                    angle_diff = np.abs(np.pi - angle_diff)
                if angle_diff > horz_angle_diff_maximum_rad:
                    continue

                # Calculate line parameters
                distances = find_closest_distance_from_points_to_line_3d(points, np.array([p1, p2]))

                # Find inliers
                inlier_mask = distances <= inlier_threshold_m
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
            line_inlier_counts.append(best_inlier_count)

            if num_lines > 1 and best_inliers_mask is not None:
                points = points[~best_inliers_mask]
                if len(points) <= 2:
                    break

        # combine lines if there z height is withing the inlier threshold
        if len(best_lines) > 1:
            combined_lines = []
            combined_inlier_counts = []
            for i, line in enumerate(best_lines):
                p1, p2 = line
                avg_height = (p1[2] + p2[2]) / 2

                for j, (cp1, cp2) in enumerate(combined_lines):
                    combined_avg_height = (cp1[2] + cp2[2]) / 2
                    if np.abs(combined_avg_height - avg_height) <= inlier_threshold_m * 2:

                        d1 = np.linalg.norm(p1 - cp1) + np.linalg.norm(p2 - cp2)
                        d2 = np.linalg.norm(p1 - cp2) + np.linalg.norm(p2 - cp1)
                        combined_inlier_count = combined_inlier_counts[j]
                        line_inlier_count = line_inlier_counts[i]
                        total_inlier_count = combined_inlier_count + line_inlier_count
                        if d1 < d2:
                            new_p1 = (cp1 * combined_inlier_count + p1 * line_inlier_count) / total_inlier_count
                            new_p2 = (cp2 * combined_inlier_count + p2 * line_inlier_count) / total_inlier_count
                        else:
                            new_p1 = (cp1 * combined_inlier_count + p2 * line_inlier_count) / total_inlier_count
                            new_p2 = (cp2 * combined_inlier_count + p1 * line_inlier_count) / total_inlier_count

                        combined_lines[j] = (new_p1, new_p2)
                        combined_inlier_counts[j] += line_inlier_counts[i]
                        break
                else:
                    combined_lines.append(line)
                    combined_inlier_counts.append(line_inlier_counts[i])

            return combined_lines, combined_inlier_counts

        return best_lines, line_inlier_counts