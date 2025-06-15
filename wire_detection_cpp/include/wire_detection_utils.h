#ifndef WIRE_DETECTION_UTILS_H
#define WIRE_DETECTION_UTILS_H

#include <tuple>
#include <vector>
#include <utility>
#include <string>
#include <limits>
#include <cmath>
#include <numeric>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

// Configuration struct to load from YAML
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

    static WireDetectionConfig fromYAML(const std::string& filepath);
};

// Finds peaks in a vector with a given height range
std::vector<int> find_peaks(const std::vector<double>& x, std::pair<double, double> height);

// Finds local maxima (midpoints, left and right edges)
std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> local_maxima_1d(const std::vector<double>& x);

// Converts peak histogram bins into wire distances, filtering by threshold
std::vector<double> peak_hist_into_wires(const std::vector<int>& hist, const std::vector<double>& bin_edges, int bin_threshold);

// Finds closest distances from 3D points to a line segment defined by two endpoints
Eigen::VectorXd find_closest_distance_from_points_to_line_3d(
    const Eigen::MatrixXd& points,         // Nx3 matrix
    const Eigen::MatrixXd& line_ends       // 2x3 matrix
);

// Computes the length of the center line across an image given its dimensions and angle
double get_length_of_center_line_across_image(int image_height, int image_width, double angle);

// Returns perpendicular angle in radians, folded to [-π/2, π/2]
double perpendicular_angle_rad(double angle_rad);

// Vectorized version of perpendicular_angle_rad for Eigen vectors
Eigen::VectorXd perpendicular_angles_rad(const Eigen::VectorXd& angles_rad);

// Fold angle from [0, 2π) to [0, π]
double fold_angle_from_0_to_pi(double angle);

// Vectorized version of fold_angle_from_0_to_pi for Eigen vectors
Eigen::VectorXd fold_angles_from_0_to_pi(const Eigen::VectorXd& angles);

// Projects an image's pixel values onto an axis defined by yaw angle, returns pairs of projections and values
std::pair<std::vector<double>, std::vector<double>> project_image_to_axis(
    const Eigen::MatrixXd& value_img, double yaw_rad
);

// Computes sine and cosine of scaled (radial) samples given a period
void circfuncs_common(const std::vector<float>& samples,
                      float period,
                      std::vector<float>& sin_samp,
                      std::vector<float>& cos_samp);

// Computes circular mean of angles in radians
float circmean(const std::vector<float>& samples,
               float high = 2.0f * M_PI,
               float low = 0.0f);

std::vector<cv::Point> get_pixels_from_line(const std::vector<cv::Vec4i>& lines, int img_width, int img_height)

#endif // WIRE_DETECTION_UTILS_H
