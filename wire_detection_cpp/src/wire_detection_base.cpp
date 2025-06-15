#include "wire_detection_base.h"
#include "wire_detection_utils.h"

WireDetector::WireDetector(const WireDetectionConfig &config,
                           const Eigen::Matrix3f &intrinsics, )
    : hough_vote_threshold(config.hough_vote_threshold),
      min_line_threshold(config.min_line_threshold),
      pixel_binning_size(config.pixel_binning_size),
      low_canny_threshold(config.low_canny_threshold),
      high_canny_threshold(config.high_canny_threshold),
      line_bin_avg_threshold_multiplier(config.line_bin_avg_threshold_multiplier),
      grad_bin_avg_threshold(config.grad_bin_avg_threshold),
      max_wire_per_roi(config.max_wire_per_roi),
      min_depth_clip(config.min_depth_clip_m),
      max_depth_clip(config.max_depth_clip_m),
      ransac_max_iters(config.ransac_max_iters),
      inlier_threshold_m(config.inlier_threshold_m),
      vert_angle_maximum_rad(config.vert_angle_maximum_rad),
      horz_angle_diff_maximum_rad(config.horz_angle_diff_maximum_rad),
      camera_intrinsics(intrinsics),
      inv_camera_intrinsics(intrinsics.inverse()),
{
}

std::tuple<cv::Mat, Eigen::Vector4i, double> WireDetector::getLineCandidates(const cv::Mat& rgb_image) {
    // 1. Get Hough lines Nx4 float matrix
    cv::Mat cartesian_lines = getHoughLines(rgb_image);
    int N = cartesian_lines.rows;

    // 2. Vectorized angle computation (atan2(dy, dx))
    Eigen::VectorXd deltas_y(N), deltas_x(N);
    for (int i = 0; i < N; i++) {
        deltas_x[i] = cartesian_lines.at<float>(i, 2) - cartesian_lines.at<float>(i, 0);
        deltas_y[i] = cartesian_lines.at<float>(i, 3) - cartesian_lines.at<float>(i, 1);
    }

    Eigen::VectorXd line_angles = (deltas_y.array() / deltas_x.array()).unaryExpr([](double) { return 0; }); // dummy init
    for (int i = 0; i < N; i++) {
        line_angles[i] = std::atan2(deltas_y[i], deltas_x[i]);
    }

    // 3. Fold angles [0, pi]
    Eigen::VectorXd folded_angles = fold_angles_from_0_to_pi(line_angles);

    // 4. Circular mean of folded angles [0, pi]
    double avg_angle = circmean(folded_angles, M_PI, 0);

    // 5. Initialize image parameters and camera rays once
    if (!initialized) {
        this->img_shape = rgb_image.size();
        this->img_height = rgb_image.rows;
        this->img_width = rgb_image.cols;
        this->cx = this->img_width / 2;
        this->cy = this->img_height / 2;
        double line_length = std::max(this->img_width, this->img_height) * 2;

        // Create meshgrid coordinates
        int num_pixels = this->img_height * this->img_width;
        Eigen::MatrixXd pixel_coords(3, num_pixels);
        Eigen::RowVectorXd x_vec = Eigen::VectorXd::LinSpaced(this->img_width, 0, this->img_width - 1).transpose();
        Eigen::RowVectorXd y_vec = Eigen::VectorXd::LinSpaced(this->img_height, 0, this->img_height - 1).transpose();

        // Repeat x_vec img_height times horizontally
        Eigen::RowVectorXd x_coords = x_vec.replicate(1, this->img_height).reshaped(1, num_pixels);

        // Repeat y_vec img_width times vertically
        Eigen::RowVectorXd y_coords(this->img_width * this->img_height);
        for (int i = 0; i < this->img_height; i++) {
            y_coords.segment(i * this->img_width, this->img_width).setConstant(i);
        }

        // Assign into pixel_coords
        pixel_coords.row(0) = x_coords;
        pixel_coords.row(1) = y_coords;
        pixel_coords.row(2).setOnes();

        // Calculate camera rays: inv_camera_intrinsics * pixel_coords
        this->camera_rays = (this->inv_camera_intrinsics * pixel_coords).transpose();  // Nx3

        this->initialized = true;
    }

    // 6. Compute center line endpoints
    int x0_avg = static_cast<int>(cx + line_length * std::cos(avg_angle));
    int y0_avg = static_cast<int>(cy + line_length * std::sin(avg_angle));
    int x1_avg = static_cast<int>(cx - line_length * std::cos(avg_angle));
    int y1_avg = static_cast<int>(cy - line_length * std::sin(avg_angle));

    Eigen::Vector4i center_line(x0_avg, y0_avg, x1_avg, y1_avg);

    return std::make_tuple(cartesian_lines, center_line, avg_angle);
}

std::tuple<
        Eigen::MatrixXi,      // wire_lines (int matrix)
        Eigen::MatrixXd,      // wire_midpoints (double matrix)
        std::vector<int>,     // hist (histogram counts)
        std::vector<double>,  // bin_edges
        double,               // bin_threshold
        Eigen::VectorXd       // wire_distances_wrt_center
    >
    get_line_instance_locations(
        const Eigen::MatrixXi& cartesian_lines,
        const Eigen::Vector4i& center_line,
        double avg_angle
    ) {
        // Calculate perp angle and max distance for bins
        double perp_angle = perpendicular_angle_rad(avg_angle);
        double image_perp_distance = get_length_of_center_line_across_image(img_height, img_width, perp_angle);

        // Create bins: from -image_perp_distance/2 to +image_perp_distance/2 stepping by pixel_binning_size
        int num_bins = static_cast<int>(std::ceil(image_perp_distance / pixel_binning_size)) + 1;
        std::vector<double> bins(num_bins);
        double start_bin = -image_perp_distance / 2.0;
        for (int i = 0; i < num_bins; ++i) {
            bins[i] = start_bin + i * pixel_binning_size;
        }

        // Compute perpendicular distances for all lines
        Eigen::VectorXd pixel_dists_wrt_center = compute_perpendicular_distance(center_line, cartesian_lines);

        // Histogram the distances into bins
        std::vector<int> hist(num_bins - 1, 0);
        for (int i = 0; i < pixel_dists_wrt_center.size(); ++i) {
            double val = pixel_dists_wrt_center(i);
            if (val < bins.front() || val > bins.back()) continue;
            auto it = std::upper_bound(bins.begin(), bins.end(), val);
            int idx = static_cast<int>(std::distance(bins.begin(), it)) - 1;
            if (idx >= 0 && idx < static_cast<int>(hist.size())) hist[idx]++;
        }

        // Calculate threshold for wire detection based on average nonzero histogram counts
        std::vector<int> hist_nonzero;
        std::copy_if(hist.begin(), hist.end(), std::back_inserter(hist_nonzero), [](int x){ return x > 0; });
        double mean_hist_nonzero = hist_nonzero.empty() ? 0.0 : (std::accumulate(hist_nonzero.begin(), hist_nonzero.end(), 0.0) / hist_nonzero.size());
        double bin_threshold = line_bin_avg_threshold_multiplier * mean_hist_nonzero;

        // Find wire distances by detecting peaks in histogram
        Eigen::VectorXd wire_distances_wrt_center = peak_hist_into_wires(hist, bins, pixel_dists_wrt_center, bin_threshold);
        wire_distances_wrt_center = -wire_distances_wrt_center;  // flip sign to match center line direction

        // Calculate offsets for wire midpoints
        double sin_offset = std::sin(avg_angle + M_PI / 2);
        double cos_offset = std::cos(avg_angle + M_PI / 2);

        // Compute wire midpoints vectorized
        Eigen::MatrixXd wire_midpoints(wire_distances_wrt_center.size(), 2);
        for (int i = 0; i < wire_distances_wrt_center.size(); ++i) {
            wire_midpoints(i, 0) = cx - wire_distances_wrt_center(i) * cos_offset;
            wire_midpoints(i, 1) = cy - wire_distances_wrt_center(i) * sin_offset;
        }

        // Compute wire line endpoints vectorized
        Eigen::VectorXd cos_avg = Eigen::VectorXd::Constant(wire_distances_wrt_center.size(), std::cos(avg_angle));
        Eigen::VectorXd sin_avg = Eigen::VectorXd::Constant(wire_distances_wrt_center.size(), std::sin(avg_angle));

        Eigen::VectorXd new_x0 = wire_midpoints.col(0).array() + line_length * cos_avg.array();
        Eigen::VectorXd new_y0 = wire_midpoints.col(1).array() + line_length * sin_avg.array();
        Eigen::VectorXd new_x1 = wire_midpoints.col(0).array() - line_length * cos_avg.array();
        Eigen::VectorXd new_y1 = wire_midpoints.col(1).array() - line_length * sin_avg.array();

        // Convert to integer wire lines
        Eigen::MatrixXi wire_lines(wire_distances_wrt_center.size(), 4);
        for (int i = 0; i < wire_distances_wrt_center.size(); ++i) {
            wire_lines(i, 0) = static_cast<int>(std::round(new_x0(i)));
            wire_lines(i, 1) = static_cast<int>(std::round(new_y0(i)));
            wire_lines(i, 2) = static_cast<int>(std::round(new_x1(i)));
            wire_lines(i, 3) = static_cast<int>(std::round(new_y1(i)));
        }

        return std::make_tuple(wire_lines, wire_midpoints, hist, bins, bin_threshold, wire_distances_wrt_center);
    }

