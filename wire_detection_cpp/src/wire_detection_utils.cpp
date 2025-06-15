#include <tuple>
#include <cstddef> // for size_t
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Core>


#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>

#include <wire_detection_utils.h>

using namespace std;

struct WireDetectionConfig {
    int hough_vote_threshold;
    int min_line_threshold;
    int pixel_binning_size;
    int low_canny_threshold;
    int high_canny_threshold;
    float line_bin_avg_threshold_multiplier;

    float grad_bin_avg_threshold;
    int max_wire_per_roi;
    float min_depth_clip_m;
    float max_depth_clip_m;

    int ransac_max_iters;
    float inlier_threshold_m;
    float vert_angle_maximum_rad;
    float horz_angle_diff_maximum_rad;

    static WireDetectionConfig fromYAML(const std::string& filepath) {
        YAML::Node config = YAML::LoadFile(filepath);
        WireDetectionConfig c;
        c.hough_vote_threshold = config["hough_vote_threshold"].as<int>();
        c.min_line_threshold = config["min_line_threshold"].as<int>();
        c.pixel_binning_size = config["pixel_binning_size"].as<int>();
        c.low_canny_threshold = config["low_canny_threshold"].as<int>();
        c.high_canny_threshold = config["high_canny_threshold"].as<int>();
        c.line_bin_avg_threshold_multiplier = config["line_bin_avg_threshold_multiplier"].as<float>();

        c.grad_bin_avg_threshold = config["grad_bin_avg_threshold"].as<float>();
        c.max_wire_per_roi = config["max_wire_per_roi"].as<int>();
        c.min_depth_clip_m = config["min_depth_clip_m"].as<float>();
        c.max_depth_clip_m = config["max_depth_clip_m"].as<float>();

        c.ransac_max_iters = config["ransac_max_iters"].as<int>();
        c.inlier_threshold_m = config["inlier_threshold_m"].as<float>();
        c.vert_angle_maximum_rad = config["vert_angle_maximum_rad"].as<float>();
        c.horz_angle_diff_maximum_rad = config["horz_angle_diff_maximum_rad"].as<float>();

        return c;
    }
};

vector<int> find_peaks(const vector<double>& x, pair<double, double> height)
{
    vector<int> peaks, left_edges, right_edges;
    tie(peaks, left_edges, right_edges) = local_maxima_1d(x);

    vector<double> peak_heights;
    peak_heights.reserve(peaks.size());
    for (int idx : peaks)
        peak_heights.push_back(x[idx]);

    double hmin = height.first;
    double hmax = height.second;
    vector<bool> keep;
    keep.reserve(peaks.size());

    for (int idx : peaks) {
        double value = x[idx];
        bool is_valid = true;
        is_valid &= (value >= hmin);
        is_valid &= (value <= hmax);
        keep.push_back(is_valid);
    }

    // Filter peaks based on height criteria
    vector<int> filtered_peaks;
    for (size_t i = 0; i < peaks.size(); ++i) {
        if (keep[i]) {
            filtered_peaks.push_back(peaks[i]);
        }
    }

    return filtered_peaks;
}

tuple<vector<int>, vector<int>, vector<int>> local_maxima_1d(const vector<double>& x) {
    vector<int> midpoints;
    vector<int> left_edges;
    vector<int> right_edges;

    const size_t size = x.size();
    if (size < 3)
        return {midpoints, left_edges, right_edges};

    size_t reserve_size = size / 2;
    midpoints.reserve(reserve_size);
    left_edges.reserve(reserve_size);
    right_edges.reserve(reserve_size);

    size_t i = 1;
    size_t i_max = size - 1;

    while (i < i_max)
    {
        if (x[i - 1] < x[i])
        {
            size_t i_ahead = i + 1;
            while (i_ahead < i_max && x[i_ahead] == x[i])
                i_ahead++;

            if (x[i_ahead] < x[i])
            {
                int left = static_cast<int>(i);
                int right = static_cast<int>(i_ahead - 1);
                int mid = (left + right) / 2;

                left_edges.push_back(left);
                right_edges.push_back(right);
                midpoints.push_back(mid);

                i = i_ahead;
                continue;
            }
        }
        ++i;
    }

    return {midpoints, left_edges, right_edges};
}


std::vector<double> peak_hist_into_wires(const std::vector<int>& hist,const std::vector<double>& bin_edges, int bin_threshold)
{
    assert(bin_edges.size() == hist.size() + 1);

    // Find peaks with counts >= bin_threshold
    std::vector<int> peaks = find_peaks(hist, bin_threshold);

    // Calculate bin centers = (bin_edges[i] + bin_edges[i+1]) / 2
    std::vector<double> bin_centers(hist.size());
    for (size_t i = 0; i < hist.size(); ++i) {
        bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2.0;
    }

    // Collect distances corresponding to peak bin centers
    std::vector<double> distances;
    distances.reserve(peaks.size());

    for (int peak_idx : peaks) {
        distances.push_back(bin_centers[peak_idx]);
    }

    return distances;
}

// points: Nx3 matrix, line_ends: 2x3 matrix
Eigen::VectorXd find_closest_distance_from_points_to_line_3d(
    const Eigen::MatrixXd& points,         // Nx3 matrix
    const Eigen::MatrixXd& line_ends       // 2x3 matrix
) {
    assert(points.cols() == 3 && "Points must be 3D");
    assert(line_ends.rows() == 2 && line_ends.cols() == 3 && "Line ends must be 2x3");

    const Eigen::RowVector3d p1 = line_ends.row(0);
    const Eigen::RowVector3d p2 = line_ends.row(1);
    const Eigen::RowVector3d line_vector = p2 - p1;

    const double line_length_squared = line_vector.squaredNorm();

    // Vector from p1 to each point
    Eigen::MatrixXd p1_to_points = points.rowwise() - p1;

    // Project each point onto the line direction
    Eigen::VectorXd t = (p1_to_points * line_vector.transpose()) / line_length_squared;

    // Clamp t to [0, 1]
    t = t.array().min(1.0).max(0.0);

    // Compute closest points on the segment
    Eigen::MatrixXd closest_points = t.replicate(1, 3).array().colwise() * line_vector.array();
    closest_points.rowwise() += p1;

    // Compute distances
    Eigen::VectorXd distances = (points - closest_points).rowwise().norm();

    return distances;
}

double get_length_of_center_line_across_image(int image_height, int image_width, double angle)
{
    assert(image_height > 0 && image_width > 0);
    angle = fold_angle_from_0_to_pi(angle);

    double cos_angle = std::cos(angle);
    double sin_angle = std::sin(angle);

    double length = std::sqrt(std::pow(image_height * sin_angle, 2) + std::pow(image_width * cos_angle, 2));
    return length;
}

double perpendicular_angle_rad(double angle_rad) {
    double perp_angle = fold_angle_from_0_to_pi(angle_rad + PI / 2);
    if (perp_angle > PI / 2)
        perp_angle -= PI;
    return perp_angle;
}

// Vectorized version of perpendicular angle
Eigen::VectorXd perpendicular_angles_rad(const Eigen::VectorXd& angles_rad) {
    return angles_rad.unaryExpr([](double angle_rad) {
        double perp_angle = fold_angles_from_0_to_pi(angle_rad + PI / 2);
        if (perp_angle > PI / 2)
            perp_angle -= PI;
        return perp_angle;
    });
}

double fold_angle_from_0_to_pi(double angle) {
    angle = std::fmod(angle, 2 * PI);
    if (angle < 0)
        angle += 2 * PI;
    return (angle > PI) ? (angle - PI) : angle;
}

// Vector version
Eigen::VectorXd fold_angles_from_0_to_pi(const Eigen::VectorXd& angles) {
    Eigen::VectorXd folded = angles.unaryExpr([](double angle) {
        angle = std::fmod(angle, 2 * PI);
        if (angle < 0)
            angle += 2 * PI;
        return (angle > PI) ? (angle - PI) : angle;
    });
    return folded;
}

std::pair<std::vector<double>, std::vector<double>> project_image_to_axis(
    const Eigen::MatrixXd& value_img, double yaw_rad)
{
    int H = value_img.rows();
    int W = value_img.cols();

    Eigen::Vector2d center(W / 2.0, H / 2.0);
    Eigen::Vector2d axis(std::cos(yaw_rad), std::sin(yaw_rad));

    std::vector<double> projections;
    std::vector<double> values;

    projections.reserve(H * W);
    values.reserve(H * W);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            Eigen::Vector2d coord(x - center[0], y - center[1]);
            double proj = coord.dot(axis);
            projections.push_back(proj);
            values.push_back(value_img(y, x));
        }
    }
    return {projections, values};
}

// Computes sin and cos of scaled (radial) samples
void circfuncs_common(const std::vector<float>& samples,
                      float period,
                      std::vector<float>& sin_samp,
                      std::vector<float>& cos_samp) {
    size_t n = samples.size();
    sin_samp.resize(n);
    cos_samp.resize(n);
    float scale = 2.0f * M_PI / period;

    for (size_t i = 0; i < n; ++i) {
        float scaled = samples[i] * scale;
        sin_samp[i] = std::sin(scaled);
        cos_samp[i] = std::cos(scaled);
    }
}

// Main function to compute circular mean
float circmean(const std::vector<float>& samples,
               float high = 2.0f * M_PI,
               float low = 0.0f) {
    size_t n = samples.size();
    if (n == 0) return std::numeric_limits<float>::quiet_NaN();

    float period = high - low;
    std::vector<float> sin_samp, cos_samp;
    circfuncs_common(samples, period, sin_samp, cos_samp);

    float sin_sum = std::accumulate(sin_samp.begin(), sin_samp.end(), 0.0f);
    float cos_sum = std::accumulate(cos_samp.begin(), cos_samp.end(), 0.0f);

    float res = std::atan2(sin_sum, cos_sum);  // result in [-π, π]
    // Map result into [0, 2π] → then scale to [low, high]
    float mean = std::fmod(res * (period / (2.0f * M_PI)) - low + period, period) + low;
    return mean;
}

std::vector<cv::Point> get_pixels_from_line(const std::vector<cv::Vec4i>& lines, int img_width, int img_height) {
    cv::Mat img_mask = cv::Mat::zeros(img_height, img_width, CV_8UC1);

    for (const auto& line : lines) {
        cv::line(img_mask, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255), 1);
    }

    std::vector<cv::Point> pixels;
    cv::findNonZero(img_mask, pixels);  // Gets all points where pixel value != 0 (==255 here)

    return pixels;
}
