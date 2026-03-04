#include <memory>
#include <Eigen/Geometry>

#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp>
#include <glim/util/config.hpp>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>

#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <gtsam_points/config.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/parallelism.hpp>

namespace glim {
DynamicObjectRejectionParamsCPU::DynamicObjectRejectionParamsCPU() {
    Config config(GlobalConfig::get_config_path("config_dynamic_object_rejection"));
    mean_difference_threshold = config.param<double>("dynamic_object_rejection", "mean_difference_threshold", 0.5);
    covariance_error_threshold = config.param<double>("dynamic_object_rejection", "covariance_error_threshold", 0.5);
    points_number_difference_threshold = config.param<int>("dynamic_object_rejection", "points_number_difference_threshold", 10);
    num_threads = config.param<int>("dynamic_object_rejection", "num_threads", 4);
    
    //this one must be the same as the one used for odometry estimation to ensure consistency in voxelization
    voxel_resolution = config.param<double>("odometry_estimation", "voxel_resolution", 0.5);
    voxel_resolution_max = config.param<double>("odometry_estimation", "voxel_resolution_max", voxel_resolution);
    voxel_resolution_dmin = config.param<double>("odometry_estimation", "voxel_resolution_dmin", 4.0);
    voxel_resolution_dmax = config.param<double>("odometry_estimation", "voxel_resolution_dmax", 12.0);

    voxelmap_levels = config.param<int>("odometry_estimation", "voxelmap_levels", 2);
    voxelmap_scaling_factor = config.param<double>("odometry_estimation", "voxelmap_scaling_factor", 2.0);

    
}


DynamicObjectRejectionParamsCPU::~DynamicObjectRejectionParamsCPU() {}

DynamicObjectRejectionCPU::DynamicObjectRejectionCPU(const DynamicObjectRejectionParamsCPU& params)
    : params_(params) {
        dynamic_voxels_indices.clear();
    }

// voxelize now uses member covariance_estimation
std::vector<gtsam_points::DynamicVoxelMapCPU::Ptr> DynamicObjectRejectionCPU::voxelize(const PreprocessedFrame::Ptr& raw_frame) {
    // for now we assume the points are already in lidar coordinate; no transform
    std::vector<Eigen::Vector4d> points_imu = raw_frame->points;

    std::vector<Eigen::Vector4d> normals;
    std::vector<Eigen::Matrix4d> covs;

    auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points_imu);
    if (!raw_frame->intensities.empty()) {
      frame->add_intensities(raw_frame->intensities);
    }
    frame->add_covs(covs);
    frame->add_normals(normals);

    std::vector<gtsam_points::DynamicVoxelMapCPU::Ptr> voxelmaps;
    double current_resolution = params_.voxel_resolution;
    for (int i = 0; i < params_.voxelmap_levels; ++i) {
        auto voxelmap = std::make_shared<gtsam_points::DynamicVoxelMapCPU>(current_resolution);
        // insert expects a const PointCloud&, so dereference the shared_ptr
        voxelmap->insert(*frame);
        voxelmaps.push_back(voxelmap);
        current_resolution *= params_.voxelmap_scaling_factor;
        if (current_resolution > params_.voxel_resolution_max) {
            current_resolution = params_.voxel_resolution_max;
        }
    }
    
    return voxelmaps;

}


std::vector<gtsam_points::DynamicVoxelMapCPU::Ptr> DynamicObjectRejectionCPU::add_odometry(
    const std::vector<gtsam_points::DynamicVoxelMapCPU::Ptr>& voxelmaps,
    const Eigen::Isometry3d& T_world_imu) {
    // std::vector<DynamicVoxelMapCPU::Ptr> updated_voxelmaps;
    // for (const auto& voxelmap : voxelmaps) {
    //     // create new map of same type/resolution
    //     auto updated_voxelmap = std::make_shared<DynamicVoxelMapCPU>(voxelmap->voxel_resolution());
    //     for (int i = 0; i < voxelmap->num_voxels(); i++) {
    //         auto& voxel = voxelmap->lookup_voxel(i);
    //         Eigen::Vector4d transformed_mean = T_world_imu * Eigen::Vector4d(voxel.mean.x(), voxel.mean.y(), voxel.mean.z(), 1.0);
    //         // Transform the covariance matrix correctly
    //         Eigen::Matrix3d R = T_world_imu.rotation();
    //         Eigen::Matrix3d transformed_cov = R * voxel.cov * R.transpose();
    //         DynamicGaussianVoxel transformed_voxel;
    //         transformed_voxel.mean = transformed_mean.head<3>();
    //         transformed_voxel.cov = transformed_cov;
    //         transformed_voxel.num_points = voxel.num_points;
    //         transformed_voxel.is_dynamic = voxel.is_dynamic;
    //         updated_voxelmap->add_voxel(transformed_voxel);
    //     }
    //     updated_voxelmaps.push_back(updated_voxelmap);
    // }
    // return updated_voxelmaps;
    return voxelmaps; // for now we skip the actual transformation for simplicity; in a real implementation we would need to transform each voxel according to the estimated pose
}

PreprocessedFrame::Ptr DynamicObjectRejectionCPU::dynamic_object_rejection(const PreprocessedFrame::Ptr frame, EstimationFrame::ConstPtr prev_frame){
    // Voxelize the current frame
    auto voxelmaps = voxelize(frame);

    // Add odometry information to the voxelmaps
    voxelmaps = add_odometry(voxelmaps, prev_frame->T_world_imu);

    // Compare the voxelmaps of the current frame with those of the previous frame to identify dynamic points
    for (int i = 0; i < static_cast<int>(voxelmaps.size()); i++)
    {
        if (prev_frame && i < static_cast<int>(prev_frame->voxelmaps.size())) {
            auto current_voxelmap = voxelmaps[i];
            // previous frame stores generic GaussianVoxelMap pointers; attempt dynamic cast
            auto prev_generic = prev_frame->voxelmaps[i];
            auto prev_voxelmap = std::dynamic_pointer_cast<gtsam_points::DynamicVoxelMapCPU>(prev_generic);
            if (!prev_voxelmap) {
                continue; // cannot perform comparison without correct type
            }
            // Compare each voxel in the current voxelmap with the corresponding voxel in the previous voxelmap
            int nvox = static_cast<int>(
                current_voxelmap->gtsam_points::IncrementalVoxelMap<gtsam_points::DynamicGaussianVoxel>::num_voxels());
            for (int j = 0; j < nvox; j++) {
                auto& current_voxel = current_voxelmap->lookup_voxel(j);
                auto& prev_voxel = prev_voxelmap->lookup_voxel(j);
                if (current_voxel.is_dynamic) {
                    double mean_diff = (current_voxel.mean - prev_voxel.mean).norm();
                    if (mean_diff < params_.mean_difference_threshold) {
                        current_voxel.is_dynamic = false;
                    }
                    double cov_diff = (current_voxel.cov - prev_voxel.cov).norm();
                    if (cov_diff < params_.covariance_error_threshold) {
                        current_voxel.is_dynamic = false;
                    }
                    // compute absolute difference safely by casting to signed type
                    long long diff_ll = static_cast<long long>(current_voxel.num_points) - static_cast<long long>(prev_voxel.num_points);
                    int points_diff = static_cast<int>(diff_ll < 0 ? -diff_ll : diff_ll);
                    if (points_diff < params_.points_number_difference_threshold) {
                        current_voxel.is_dynamic = false;
                    }
                } else {
                    auto& current_voxel = current_voxelmap->lookup_voxel(j);
                    current_voxel.is_dynamic = false;
                }
            }
        }
    }
    last_voxelmap = voxelmaps.back();
    gtsam_points::PointCloudCPU::Ptr new_frame = std::make_shared<gtsam_points::PointCloudCPU>();
    // iterate using the IncrementalVoxelMap candidate to disambiguate
    int nvox = static_cast<int>(
        last_voxelmap->gtsam_points::IncrementalVoxelMap<gtsam_points::DynamicGaussianVoxel>::num_voxels());
    for (int i = 0; i < nvox; i++)
    {
        auto& current_voxel = last_voxelmap->lookup_voxel(i);
        if (current_voxel.is_dynamic) {
            dynamic_voxels_indices.push_back(i);
        } else {
            // Add points from this voxel to the new frame
            new_frame->add_points(current_voxel.voxel_points);
            if (frame->intensities.size()) {
                new_frame->add_intensities(current_voxel.voxel_intensities);
            }
            new_frame->add_times(current_voxel.voxel_times);
        }
    }
    PreprocessedFrame::Ptr dynamic_rejection_frame(new PreprocessedFrame);
    dynamic_rejection_frame->stamp = frame->stamp;
    dynamic_rejection_frame->scan_end_time = frame->scan_end_time;
    if (new_frame->times) {
        dynamic_rejection_frame->times.assign(new_frame->times, new_frame->times + new_frame->size());
    }
    dynamic_rejection_frame->points.assign(new_frame->points, new_frame->points + new_frame->size());
    if (new_frame->intensities) {
        dynamic_rejection_frame->intensities.assign(new_frame->intensities, new_frame->intensities + new_frame->size());
    }
    dynamic_rejection_frame->k_neighbors = frame->k_neighbors;
    dynamic_rejection_frame->neighbors = find_neighbors(new_frame->points, new_frame->size(), frame->k_neighbors);
    
    return dynamic_rejection_frame;
}


std::vector<int> DynamicObjectRejectionCPU::find_neighbors(const Eigen::Vector4d* points, const int num_points, const int k) const {
  gtsam_points::KdTree tree(points, num_points);

  std::vector<int> neighbors(num_points * k);

  const auto perpoint_task = [&](int i) {
    std::vector<size_t> k_indices(k, i);
    std::vector<double> k_sq_dists(k);
    size_t num_found = tree.knn_search(points[i].data(), k, k_indices.data(), k_sq_dists.data());
    std::copy(k_indices.begin(), k_indices.begin() + num_found, neighbors.begin() + i * k);
  };

  if (gtsam_points::is_omp_default()) {
#pragma omp parallel for num_threads(params_.num_threads) schedule(guided, 8)
    for (int i = 0; i < num_points; i++) {
      perpoint_task(i);
    }
  } else {
#ifdef GTSAM_POINTS_USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, num_points, 8), [&](const tbb::blocked_range<int>& range) {
      for (int i = range.begin(); i < range.end(); i++) {
        perpoint_task(i);
      }
    });
#else
    std::cerr << "error : TBB is not enabled" << std::endl;
    abort();
#endif
  }

  return neighbors;
}

}

