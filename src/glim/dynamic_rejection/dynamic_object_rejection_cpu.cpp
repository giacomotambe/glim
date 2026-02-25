#include <Eigen/Geometry>

#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>

namespace dynamic_glim {
DynamicObjectRejectionParams::DynamicObjectRejectionParams() {
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

    dynamic_voxels_indices.clear();
}


DynamicObjectRejectionParams::~DynamicObjectRejectionParams() {}

std::vector<DynamicVoxelMap::Ptr> voxelize(const PreprocessedFrame::Ptr& raw_frame){
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

    std::vector<DynamicVoxelMap::Ptr> voxelmaps;
    double current_resolution = params_.voxel_resolution;
    for (int i = 0; i < params_.voxelmap_levels; ++i) {
        // #ifdef GTSAM_POINTS_USE_CUDA
        //     auto voxelmap = std::make_shared<gtsam_points::DynamicGaussianVoxelMapGPU>(current_resolution, 8192 * 2, 10, 1e-3, *stream);
        // #else
        //     auto voxelmap = std::make_shared<gtsam_points::DynamicGaussianVoxelMapCPU>(current_resolution);
        // #endif
        auto voxelmap = std::make_shared<DynamicVoxelMapCPU>(current_resolution);
        voxelmap->insert(frame);
        voxelmaps.push_back(voxelmap);
        current_resolution *= params_.voxelmap_scaling_factor;
        if (current_resolution > params_.voxel_resolution_max) {
            current_resolution = params_.voxel_resolution_max;
        }
    }
    
    return voxelmaps;

}

std::vector<gtsam_points::GaussianVoxelMap::Ptr> add_odometry(const std::vector<gtsam_points::GaussianVoxelMap::Ptr>& voxelmaps, const Eigen::Isometry3d& T_world_imu){
    std::vector<gtsam_points::GaussianVoxelMap::Ptr> updated_voxelmaps;
    for (const auto& voxelmap : voxelmaps) {
        auto updated_voxelmap = std::make_shared<gtsam_points::GaussianVoxelMapCPU>(voxelmap->resolution());
        for (int i = 0; i < voxelmap->num_voxel(); i++) {
            auto voxel = voxelmap->lookup_voxel(i);
            Eigen::Vector4d transformed_mean = T_world_imu * Eigen::Vector4d(voxel.mean.x(), voxel.mean.y(), voxel.mean.z(), 1.0);
            // Here we should also transform the covariance and the points in the voxel, but for simplicity we will just update the mean
            DynamicGaussianVoxel transformed_voxel;
            transformed_voxel.mean = transformed_mean.head<3>();
            transformed_voxel.cov = voxel.cov; // This is not correct, but we would need to apply the appropriate transformation to the covariance matrix
            transformed_voxel.num_points = voxel.num_points;
            transformed_voxel.is_dynamic = voxel.is_dynamic;
            updated_voxelmap->add_voxel(transformed_voxel);
        }
        updated_voxelmaps.push_back(updated_voxelmap);
    }
    return updated_voxelmaps;
}

PreprocessedFrame::Ptr dynamic_object_rejection(const PreprocessedFrame::Ptr frame, EstimationFrame::ConstPtr prev_frame){
    // Voxelize the current frame
    auto voxelmaps = voxelize(frame);
    // Add odometry information to the voxelmaps
    //voxelmaps = add_odometry(voxelmaps, frame->T_world_imu);

    // Compare the voxelmaps of the current frame with those of the previous frame to identify dynamic points
    for (int i = 0; i < voxelmaps.size(); i++)
    {
        auto prev_voxelmap = prev_frame->voxelmaps[i];
        if (prev_frame && i < prev_frame->voxelmaps.size()) {
            auto current_voxelmap = voxelmaps[i];
            // Compare each voxel in the current voxelmap with the corresponding voxel in the previous voxelmap
            for (int j = 0; j < current_voxelmap->num_voxel(); j++) {
                if(current_voxel.is_dynamic){
                    auto current_voxel = current_voxelmap->lookup_voxel(j);
                    auto prev_voxel = prev_voxelmap->lookup_voxel(j);
                    
                    // Check mean difference
                    double mean_diff = (current_voxel.mean - prev_voxel.mean).norm();
                    if (mean_diff < params_.mean_difference_threshold) {
                        // Mark this voxel as dynamic
                        current_voxel.is_dynamic = false;
                    }

                    // Check covariance difference
                    double cov_diff = (current_voxel.cov - prev_voxel.cov).norm();
                    if (cov_diff < params_.covariance_error_threshold) {
                        // Mark this voxel as dynamic
                        current_voxel.is_dynamic = false;
                    }

                    // Check points number difference
                    int points_diff = std::abs(current_voxel.num_points - prev_voxel.num_points);
                    if (points_diff < params_.points_number_difference_threshold) {
                        // Mark this voxel as dynamic
                        current_voxel.is_dynamic = false;
                    }
                }
            }
        }
    }
    last_voxelmap=voxelmaps.back();
    gtsam_points::PointCloudCPU::Ptr new_frame = std::make_shared<gtsam_points::PointCloudCPU>();
    std::vector<Eigen::Vector4d> temp_point; 
    for (int i = 0; i < last_voxelmap->num_voxel(); i++)
    {
        auto current_voxel = last_voxelmap->lookup_voxel(i);
        if(current_voxel.is_dynamic){
            dynamic_voxels_indices.push_back(i);
        }else{
            // Add points from this voxel to the new frame

            new_frame->add_points(current_voxel->voxel_points);
            if (frame->intensities.size()) {
                new_frame->add_intensities(current_voxel->voxel_intensities);
            }
            new_frame->add_times(current_voxel->voxel_times);
        }
    }
    PreprocessedFrame::Ptr dynamic_rejection_frame(new PreprocessedFrame);
    dynamic_rejection_frame->stamp = frame->stamp;
    dynamic_rejection_frame->scan_end_time = frame->scan_end_time;
    dynamic_rejection_frame->times = new_frame->times; // You might want to filter these as well based on the dynamic points
    dynamic_rejection_frame->points = new_frame->points;
    if (new_frame->intensities.size()) {
        dynamic_rejection_frame->intensities = new_frame->intensities;
    }
    dynamic_rejection_frame->k_neighbors = frame->k_neighbors;
    dynamic_rejection_frame->neighbors = find_neighbors(new_frame->points, new_frame->size(), frame->k_neighbors);
    


    return dynamic_rejection_frame;
}


}

