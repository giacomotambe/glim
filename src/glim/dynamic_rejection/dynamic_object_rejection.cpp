#include <Eigen/Geometry>

#include <glim/dynamic_rejection/dynamic_object_rejection.hpp>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/type/gaussian_voxelmap_gpu.hpp>
#include <gtsam_points/type/gaussian_voxelmap_cpu.hpp>


namespace glim {
DynamicObjectRecognitionParams::DynamicObjectRecognitionParams() {
    Config config(GlobalConfig::get_config_path("config_odometry"));
    mean_difference_threshold = config.param<double>("dynamic_object_rejection", "mean_difference_threshold", 0.5);
    covariance_error_threshold = config.param<double>("dynamic_object_rejection", "covariance_error_threshold", 0.5);
    points_number_difference_treshhold = config.param<int>("dynamic_object_rejection", "points_number_difference_treshhold", 10);

    //this one must be the same as the one used for odometry estimation to ensure consistency in voxelization
    voxel_resolution = config.param<double>("odometry_estimation", "voxel_resolution", 0.5);
    voxel_resolution_max = config.param<double>("odometry_estimation", "voxel_resolution_max", voxel_resolution);
    voxel_resolution_dmin = config.param<double>("odometry_estimation", "voxel_resolution_dmin", 4.0);
    voxel_resolution_dmax = config.param<double>("odometry_estimation", "voxel_resolution_dmax", 12.0);

    voxelmap_levels = config.param<int>("odometry_estimation", "voxelmap_levels", 2);
    voxelmap_scaling_factor = config.param<double>("odometry_estimation", "voxelmap_scaling_factor", 2.0);

}


DynamicObjectRecognitionParams::~DynamicObjectRecognitionParams() {}

std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxelize(const PreprocessedFrame::Ptr& raw_frame){
    std::vector<Eigen::Vector4d> points_imu(raw_frame->size());
    for (int i = 0; i < raw_frame->size(); i++) {
      points_imu[i] = T_imu_lidar * raw_frame->points[i];
    }

    std::vector<Eigen::Vector4d> normals;
    std::vector<Eigen::Matrix4d> covs;
    covariance_estimation->estimate(points_imu, raw_frame->neighbors, normals, covs);

    auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points_imu);
    if (raw_frame->intensities.size()) {
      frame->add_intensities(raw_frame->intensities);
    }
    frame->add_covs(covs);
    frame->add_normals(normals);

    std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxelmaps;
    double current_resolution = params_.voxel_resolution;
    for (int i = 0; i < params_.voxelmap_levels; ++i) {
        auto voxelmap = std::make_shared<gtsam_points::GaussianVoxelMap>(current_resolution);
        voxelmap->insert(frame);
        voxelmaps.push_back(voxelmap);
        current_resolution *= params_.voxelmap_scaling_factor;
        if (current_resolution > params_.voxel_resolution_max) {
            current_resolution = params_.voxel_resolution_max;
        }
    }
    return voxelmaps;
}

std::vector<gtsam_points::GaussianVoxelMap::Ptr> add_odometry(const std::vector<gtsam_points::GaussianVoxelMap::Ptr>& voxelmaps){
    // This function would apply the necessary transformations to the voxelmaps based on the estimated odometry. 
    // The specific implementation would depend on how the odometry information is represented and how it should be applied to the voxelmaps.
    // For example, if you have an estimated pose for the current frame, you could transform the voxelmaps accordingly to align them with a common reference frame.
}

PreprocessedFrame::Ptr dynamic_object_rejection(const PreprocessedFrame::Ptr frame, EstimationFrame::ConstPtr prev_frame){
    // Voxelize the current frame
    auto voxelmaps = voxelize(frame);
    // Add odometry information to the voxelmaps
    auto prev_voxelmaps = prev_frame->voxelmaps[i];
    add_odometry(voxelmaps);
    // Compare the voxelmaps of the current frame with those of the previous frame to identify dynamic points
    for (int i = 0; i < voxelmaps.size(); i++)
    {
        if (prev_frame && i < prev_frame->voxelmaps.size()) {
            auto current_voxelmap = voxelmaps[i];
            

            // Compare each voxel in the current voxelmap with the corresponding voxel in the previous voxelmap
            for (const auto& current_voxel : current_voxelmap->voxels()) {
                auto prev_voxel = prev_voxelmap->get_voxel(current_voxel.first);
                if (prev_voxel) {
                    // Check mean difference
                    double mean_diff = (current_voxel.second.mean - prev_voxel->mean).norm();
                    if (mean_diff > params_.mean_difference_threshold) {
                        // Mark this voxel as dynamic
                        current_voxel.second.is_dynamic = true;
                    }

                    // Check covariance difference
                    double cov_diff = (current_voxel.second.cov - prev_voxel->cov).norm();
                    if (cov_diff > params_.covariance_error_threshold) {
                        // Mark this voxel as dynamic
                        current_voxel.second.is_dynamic = true;
                    }

                    // Check points number difference
                    int points_diff = std::abs(static_cast<int>(current_voxel.second.points.size()) - static_cast<int>(prev_voxel->points.size()));
                    if (points_diff > params_.points_number_difference_treshhold) {
                        // Mark this voxel as dynamic
                        current_voxel.second.is_dynamic = true;
                    }
                }
            }
        }
    }
    
    // This is a placeholder for the actual implementation of the comparison logic, which would involve checking the mean and covariance differences, as well as the number of points in each voxel, against the specified thresholds.

    // For now, we will just return the input frame without any modifications.
    return frame;
}


}

