#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <optional>

class WireDetector
{
public:
    WireDetector(const WireDetectionConfig &config,
                 const Eigen::Matrix3f &camera_intrinsics,)

    virtual ~WireDetector() = default;

    // Virtual methods to be implemented by subclasses
    virtual std::vector<cv::Vec4f> getHoughLines(const cv::Mat &rgb_image) = 0;
    virtual void getXYDepthGradients(const cv::Mat &depth_image) = 0;

protected:
    // Config params
    int hough_vote_threshold;
    int min_line_threshold;
    int pixel_binning_size;
    int low_canny_threshold;
    int high_canny_threshold;
    float line_bin_avg_threshold_multiplier;

    float grad_bin_avg_threshold;
    int max_wire_per_roi;
    float min_depth_clip;
    float max_depth_clip;

    int ransac_max_iters;
    float inlier_threshold_m;
    float vert_angle_maximum_rad;
    float horz_angle_diff_maximum_rad;

    bool initialized = false;

    // Camera intrinsics
    Eigen::Matrix3f camera_intrinsics;
    Eigen::Matrix3f inv_camera_intrinsics;

    // Image parameters
    int img_height;
    int img_width;
    cv::Size img_shape;
    float cx;
    float cy;
    float line_length;
    Eigen::MatrixXf camera_rays;
};