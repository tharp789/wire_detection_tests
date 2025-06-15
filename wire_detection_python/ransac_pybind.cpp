#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h> 
#include <pybind11/chrono.h>

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace py = pybind11;
using Eigen::Vector3d;
using Eigen::MatrixXd;
using std::vector;

// Fold angle to [0, pi]
double fold_angle_from_0_to_pi(double angle) {
    angle = std::fmod(angle, 2 * M_PI);
    if (angle < 0)
        angle += 2 * M_PI;
    return (angle > M_PI) ? (angle - M_PI) : angle;
}

// Calculate closest distances from points to line segment defined by two points p1, p2
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
    Eigen::MatrixXd closest_points = t.replicate(1, 3).array() * line_vector.replicate(t.rows(), 1).array();
    closest_points.rowwise() += p1;

    // Compute distances
    Eigen::VectorXd distances = (points - closest_points).rowwise().norm();

    return distances;
}

std::tuple<vector<std::pair<Vector3d, Vector3d>>, vector<int>>
ransac_line_fitting(
    Eigen::MatrixXd points,
    double avg_angle,
    int num_lines,
    int num_iterations,
    double inlier_threshold,
    double vert_angle_maximum_rad,
    double horz_angle_diff_maximum_rad)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, points.rows() - 1);

    avg_angle = fold_angle_from_0_to_pi(avg_angle);

    vector<std::pair<Vector3d, Vector3d>> best_lines;
    vector<int> line_inlier_counts;

    for (int line_i = 0; line_i < num_lines; ++line_i) {
        Eigen::VectorXi best_inliers_mask(points.rows());
        best_inliers_mask.setZero();
        int best_inlier_count = 0;
        std::pair<Vector3d, Vector3d> best_line;

        for (int iter = 0; iter < num_iterations; ++iter) {
            if (points.rows() < 2) break;

            // Sample two unique indices
            int idx1 = dist(rng);
            int idx2;
            do {
                idx2 = dist(rng);
            } while (idx2 == idx1);

            Vector3d p1 = points.row(idx1);
            Vector3d p2 = points.row(idx2);

            // Compute pitch angle
            double pitch_angle = std::atan2(std::abs(p2.z() - p1.z()), (p2.head<2>() - p1.head<2>()).norm());
            pitch_angle = fold_angle_from_0_to_pi(pitch_angle);
            if (pitch_angle > vert_angle_maximum_rad && pitch_angle < M_PI - vert_angle_maximum_rad) {
                continue;
            }

            // Compute yaw angle
            double yaw_angle = std::atan2(p2.y() - p1.y(), p2.x() - p1.x());
            yaw_angle = fold_angle_from_0_to_pi(yaw_angle);

            double angle_diff = std::abs(yaw_angle - avg_angle);
            if (angle_diff > M_PI / 2) {
                angle_diff = std::abs(M_PI - angle_diff);
            }
            if (angle_diff > horz_angle_diff_maximum_rad) {
                continue;
            }

            // Calculate distances
            Eigen::Matrix<double, 2, 3> line_points;
            line_points.row(0) = p1;
            line_points.row(1) = p2;
            Eigen::VectorXd distances = find_closest_distance_from_points_to_line_3d(points, line_points);

            // Find inliers
            Eigen::ArrayXd dist_array = distances.array();

            // Create a boolean mask array where distance <= threshold
            Eigen::Array<bool, Eigen::Dynamic, 1> bool_mask = (dist_array <= inlier_threshold);

            // Convert boolean mask to integer mask (1 or 0)
            Eigen::VectorXi inlier_mask = bool_mask.cast<int>();

            // Count the number of inliers (sum of ones)
            int inlier_count = inlier_mask.sum();

            if (inlier_count > best_inlier_count) {
                best_inlier_count = inlier_count;
                best_line = {p1, p2};
                best_inliers_mask = inlier_mask;
            }
        }

        if (best_inlier_count == 0) {
            break;
        }

        best_lines.push_back(best_line);
        line_inlier_counts.push_back(best_inlier_count);

        // Remove inliers
        if (num_lines > 1) {
            vector<int> outlier_indices;
            for (int i = 0; i < points.rows(); ++i) {
                if (best_inliers_mask(i) == 0) {
                    outlier_indices.push_back(i);
                }
            }

            if ((int)outlier_indices.size() <= 2) {
                break;
            }

            Eigen::MatrixXd filtered_points(outlier_indices.size(), 3);
            for (size_t i = 0; i < outlier_indices.size(); ++i) {
                filtered_points.row(i) = points.row(outlier_indices[i]);
            }
            points = filtered_points;
            dist = std::uniform_int_distribution<>(0, points.rows() - 1);
        }
    }

    // Combine lines if their avg z height is within threshold
    if (best_lines.size() > 1) {
        vector<std::pair<Vector3d, Vector3d>> combined_lines;
        vector<int> combined_inlier_counts;

        for (size_t i = 0; i < best_lines.size(); ++i) {
            const auto& line = best_lines[i];
            double avg_height = (line.first.z() + line.second.z()) / 2.0;

            bool merged = false;
            for (size_t j = 0; j < combined_lines.size(); ++j) {
                const auto& c_line = combined_lines[j];
                double combined_avg_height = (c_line.first.z() + c_line.second.z()) / 2.0;
                if (std::abs(combined_avg_height - avg_height) <= inlier_threshold * 2) {
                    // Check distances to decide merging
                    double d1 = (line.first - c_line.first).norm() + (line.second - c_line.second).norm();
                    double d2 = (line.first - c_line.second).norm() + (line.second - c_line.first).norm();

                    int combined_count = combined_inlier_counts[j];
                    int current_count = line_inlier_counts[i];
                    int total_count = combined_count + current_count;

                    Vector3d new_p1, new_p2;
                    if (d1 < d2) {
                        new_p1 = (c_line.first * combined_count + line.first * current_count) / total_count;
                        new_p2 = (c_line.second * combined_count + line.second * current_count) / total_count;
                    }
                    else {
                        new_p1 = (c_line.first * combined_count + line.second * current_count) / total_count;
                        new_p2 = (c_line.second * combined_count + line.first * current_count) / total_count;
                    }

                    combined_lines[j] = {new_p1, new_p2};
                    combined_inlier_counts[j] += current_count;
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                combined_lines.push_back(line);
                combined_inlier_counts.push_back(line_inlier_counts[i]);
            }
        }
        return {combined_lines, combined_inlier_counts};
    }

    return {best_lines, line_inlier_counts};
}

PYBIND11_MODULE(ransac_cpp, m) {
    m.doc() = "RANSAC line fitting C++ binding";

    m.def("ransac_line_fitting", &ransac_line_fitting,
          py::arg("points"),
          py::arg("avg_angle"),
          py::arg("num_lines"),
          py::arg("num_iterations"),
          py::arg("inlier_threshold"),
          py::arg("vert_angle_maximum_rad"),
          py::arg("horz_angle_diff_maximum_rad"));
}
