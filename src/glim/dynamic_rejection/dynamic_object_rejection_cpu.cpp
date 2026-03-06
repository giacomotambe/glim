#include <memory>
#include <algorithm>
#include <cmath>
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
    spdlog::info("[dynamic_rejection] DynamicObjectRejectionParamsCPU::DynamicObjectRejectionParamsCPU begin");
    Config config(GlobalConfig::get_config_path("config_dynamic_object_rejection"));
    mean_difference_threshold = config.param<double>("dynamic_object_rejection", "mean_difference_threshold", 0.5);
    covariance_error_threshold = config.param<double>("dynamic_object_rejection", "covariance_error_threshold", 0.5);
    points_number_difference_threshold = config.param<int>("dynamic_object_rejection", "points_number_difference_threshold", 10);
    mahalanobis_distance_threshold = config.param<double>("dynamic_object_rejection", "mahalanobis_distance_threshold", 5.0);
    num_threads = config.param<int>("dynamic_object_rejection", "num_threads", 4);
    
    //this one must be the same as the one used for odometry estimation to ensure consistency in voxelization
    voxel_resolution = config.param<double>("odometry_estimation", "voxel_resolution", 0.5);
    voxel_resolution_max = config.param<double>("odometry_estimation", "voxel_resolution_max", voxel_resolution);
    voxel_resolution_dmin = config.param<double>("odometry_estimation", "voxel_resolution_dmin", 4.0);
    voxel_resolution_dmax = config.param<double>("odometry_estimation", "voxel_resolution_dmax", 12.0);

    voxelmap_levels = config.param<int>("odometry_estimation", "voxelmap_levels", 2);
    voxelmap_scaling_factor = config.param<double>("odometry_estimation", "voxelmap_scaling_factor", 2.0);

    spdlog::info(
        "[dynamic_rejection] params loaded: mean_thr={} cov_thr={} points_thr={} mahal_thr={} threads={} voxel_res={} voxel_res_max={} levels={} scale={}",
        mean_difference_threshold,
        covariance_error_threshold,
        points_number_difference_threshold,
        mahalanobis_distance_threshold,
        num_threads,
        voxel_resolution,
        voxel_resolution_max,
        voxelmap_levels,
        voxelmap_scaling_factor);
}


DynamicObjectRejectionParamsCPU::~DynamicObjectRejectionParamsCPU() {
    spdlog::info("[dynamic_rejection] DynamicObjectRejectionParamsCPU::~DynamicObjectRejectionParamsCPU");
}

DynamicObjectRejectionCPU::DynamicObjectRejectionCPU(const DynamicObjectRejectionParamsCPU& params)
    : params_(params) {
        dynamic_voxels_indices.clear();
        covariance_estimation.reset(new CloudCovarianceEstimation(params_.num_threads));
        spdlog::info("[dynamic_rejection] DynamicObjectRejectionCPU::DynamicObjectRejectionCPU");
    }



std::vector<gtsam_points::DynamicVoxelMapCPU::Ptr> DynamicObjectRejectionCPU::add_odometry(
    const std::vector<gtsam_points::DynamicVoxelMapCPU::Ptr>& voxelmaps,
    const Eigen::Isometry3d& T_world_imu) {
    spdlog::info("[dynamic_rejection] add_odometry: voxelmaps={} t=({}, {}, {})",
                  voxelmaps.size(),
                  T_world_imu.translation().x(),
                  T_world_imu.translation().y(),
                  T_world_imu.translation().z());
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
    spdlog::info("[dynamic_rejection] dynamic_object_rejection begin: frame={} prev_frame={}", static_cast<bool>(frame), static_cast<bool>(prev_frame));
    if (!frame) {
        spdlog::warn("[dynamic_rejection] dynamic_object_rejection: null input frame");
        return nullptr;
    }

    if (!prev_frame) {
        spdlog::info("[dynamic_rejection] dynamic_object_rejection: no prev_frame, use last_voxelmaps");
    }

    std::vector<Eigen::Vector4d> frame_normals;
    std::vector<Eigen::Matrix4d> frame_covs;
    
    
    spdlog::info("[dynamic_rejection] estimating covariance for {} points", frame->points.size());
    covariance_estimation->estimate(frame->points, frame->neighbors, frame_normals, frame_covs);
    spdlog::info("[dynamic_rejection] covariance estimation done: normals={} covs={}", frame_normals.size(), frame_covs.size());
    //build a point cloud from the preprocessed frame to insert into the voxelmap
    
    auto point_cloud = std::make_shared<gtsam_points::PointCloudCPU>(frame->points);
    point_cloud->add_intensities(frame->intensities);
    point_cloud->add_times(frame->times);
    point_cloud->add_covs(frame_covs);
    point_cloud->add_normals(frame_normals);
    
    auto voxelmap = std::make_shared<gtsam_points::DynamicVoxelMapCPU>(params_.voxel_resolution);
    
    
    voxelmap->insert(*point_cloud);

    
    // Compare the voxelmaps of the current frame with those of the previous frame to identify dynamic points
    // Use internally stored last_voxelmaps instead of dynamic_cast from prev_frame
    if (!last_voxelmap) {
        spdlog::info("[dynamic_rejection] last_voxelmaps empty, skip comparison (first frame)");
        
        // Ritorna frame senza filtro dinamico se è il primo frame
        dynamic_voxels_indices.clear();
        last_voxelmap = voxelmap;
        
        return frame;
    }
        
    recursive_level = 1; // reset recursive level for new frame
    dynamic_voxels_indices.clear();
    static_points.clear();
    static_intensities.clear();
    static_times.clear();
    auto update_voxelmap = dynamic_object_recognition(voxelmap, last_voxelmap);

    
    
    // Store all voxelmaps for next frame comparison (no dynamic_cast needed)
    last_voxelmap = voxelmap;
    spdlog::info("[dynamic_rejection] stored voxelmap for next frame");
    
    // Create point cloud with all accumulated points
    gtsam_points::PointCloudCPU::Ptr new_frame = std::make_shared<gtsam_points::PointCloudCPU>();
    if (!static_points.empty()) {
        new_frame->add_points(static_points);
        if (!static_intensities.empty()) {
            new_frame->add_intensities(static_intensities);
        }
        if (!static_times.empty()) {
            new_frame->add_times(static_times);
        }
    }
    PreprocessedFrame::Ptr dynamic_rejection_frame(new PreprocessedFrame);
    dynamic_rejection_frame->stamp = frame->stamp;
    dynamic_rejection_frame->scan_end_time = frame->scan_end_time;
    if (new_frame->times) {
        dynamic_rejection_frame->times.assign(new_frame->times, new_frame->times + new_frame->size());
    }
    if (new_frame->size() > 0) {
        dynamic_rejection_frame->points.assign(new_frame->points, new_frame->points + new_frame->size());
    }
    if (new_frame->intensities) {
        dynamic_rejection_frame->intensities.assign(new_frame->intensities, new_frame->intensities + new_frame->size());
    }
    dynamic_rejection_frame->k_neighbors = frame->k_neighbors;
    if (new_frame->size() > 0 && frame->k_neighbors > 0) {
        dynamic_rejection_frame->neighbors = find_neighbors(new_frame->points, new_frame->size(), frame->k_neighbors);
    }
    spdlog::info("[dynamic_rejection] frame prepared: input_points={} output_points={} dynamic_voxels={} neighbors={}",
                  frame->points.size(),
                  dynamic_rejection_frame->points.size(),
                  dynamic_voxels_indices.size(),
                  dynamic_rejection_frame->neighbors.size());
    return dynamic_rejection_frame;
}



gtsam_points::DynamicVoxelMapCPU::Ptr DynamicObjectRejectionCPU::dynamic_object_recognition(gtsam_points::DynamicVoxelMapCPU::Ptr current_voxelmap, gtsam_points::DynamicVoxelMapCPU::Ptr prev_voxelmap) {
    // Compare each voxel in the current voxelmap with the corresponding voxel in the previous voxelmap
    int nvox = static_cast<int>(
        current_voxelmap->gtsam_points::IncrementalVoxelMap<gtsam_points::DynamicGaussianVoxel>::num_voxels());
    const int prev_nvox = static_cast<int>(
        prev_voxelmap->gtsam_points::IncrementalVoxelMap<gtsam_points::DynamicGaussianVoxel>::num_voxels());
    const int compare_nvox = std::min(nvox, prev_nvox);


    for (int j = 0; j < compare_nvox; j++) {
        auto& current_voxel = current_voxelmap->lookup_voxel(j);
        // auto coord = current_voxelmap->voxel_coord(current_voxel.mean);
        // int voxel_index = current_voxelmap->lookup_voxel_index(coord);
        // auto& prev_voxel = prev_voxelmap->lookup_voxel(voxel_index);
        auto& prev_voxel = prev_voxelmap->lookup_voxel(j);
        double mean_diff = (current_voxel.mean - prev_voxel.mean).norm();
        if (mean_diff > params_.mean_difference_threshold) {
            current_voxel.is_dynamic = true;
        }

        
        double cov_diff = (current_voxel.cov - prev_voxel.cov).norm();
        if (cov_diff > params_.covariance_error_threshold) {
            current_voxel.is_dynamic = true;    
        }

        // // Mahalanobis distance check (robust to singular/invalid covariance)
        // double mahalanobis_dist = 0.0;
        // bool mahalanobis_valid = false;
        // const Eigen::Vector3d mean_delta3 = (current_voxel.mean - prev_voxel.mean).head<3>();
        // Eigen::Matrix3d cov3 = prev_voxel.cov.topLeftCorner<3, 3>();

        // if (mean_delta3.allFinite() && cov3.allFinite()) {
        //     cov3 = 0.5 * (cov3 + cov3.transpose());
        //     cov3.diagonal().array() += 1e-6;

        //     const Eigen::LDLT<Eigen::Matrix3d> ldlt(cov3);
        //     if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
        //         const Eigen::Vector3d solved = ldlt.solve(mean_delta3);
        //         if (solved.allFinite()) {
        //             const double quad = mean_delta3.dot(solved);
        //             if (std::isfinite(quad) && quad >= 0.0) {
        //                 mahalanobis_dist = std::sqrt(quad);
        //                 mahalanobis_valid = std::isfinite(mahalanobis_dist);
        //             }
        //         }
        //     }
        // }

        

        // if (mahalanobis_dist > params_.mahalanobis_distance_threshold) {
        //     current_voxel.is_dynamic = true;
        // }

        // compute absolute difference safely by casting to signed type
        if(current_voxel.num_points > 50 && prev_voxel.num_points > 50) {
            long long curr_num = static_cast<long long>(current_voxel.num_points);
            long long prev_num = static_cast<long long>(prev_voxel.num_points);
            double points_diff_percent = 100.0 * std::abs(curr_num - prev_num) / static_cast<double>(prev_num);
            if (points_diff_percent > params_.points_number_difference_threshold) {
                current_voxel.is_dynamic = true;  
            }
        } 
        
        if(recursive_level < params_.voxelmap_levels && current_voxel.is_dynamic) {
            current_voxel.finest_voxelmap=std::make_shared<gtsam_points::DynamicVoxelMapCPU>(current_voxelmap->voxel_resolution()*0.5);
            current_voxel.finest_voxelmap->insert(*current_voxel.voxel_point_cloud);

            prev_voxel.finest_voxelmap=std::make_shared<gtsam_points::DynamicVoxelMapCPU>(current_voxelmap->voxel_resolution()*0.5);
            prev_voxel.finest_voxelmap->insert(*prev_voxel.voxel_point_cloud);
            recursive_level++;
            dynamic_object_recognition(current_voxel.finest_voxelmap, prev_voxel.finest_voxelmap);
        }  
        if(!current_voxel.is_dynamic){
            // accumulate static points for output frame
            static_points.insert(static_points.end(), current_voxel.voxel_points.begin(), current_voxel.voxel_points.end());
            static_intensities.insert(static_intensities.end(), current_voxel.voxel_intensities.begin(), current_voxel.voxel_intensities.end());
            static_times.insert(static_times.end(), current_voxel.voxel_times.begin(), current_voxel.voxel_times.end());
        }
        recursive_level = 1; // reset recursive level for next voxel
        
    }
    return current_voxelmap;
}




std::vector<int> DynamicObjectRejectionCPU::find_neighbors(const Eigen::Vector4d* points, const int num_points, const int k) const {
    spdlog::info("[dynamic_rejection] find_neighbors begin: points_ptr={} num_points={} k={}", static_cast<const void*>(points), num_points, k);
    if (!points || num_points <= 0 || k <= 0) {
        spdlog::info("[dynamic_rejection] find_neighbors bypass: invalid input");
        return {};
    }

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

    spdlog::info("[dynamic_rejection] find_neighbors end: neighbors={}", neighbors.size());
    return neighbors;
}

}

