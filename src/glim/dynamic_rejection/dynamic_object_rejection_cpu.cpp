#include <memory>
#include <algorithm>
#include <cmath>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp>
#include <glim/dynamic_rejection/transformation_kalman_filter.hpp>
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
    spdlog::debug("[dynamic_rejection] DynamicObjectRejectionParamsCPU::DynamicObjectRejectionParamsCPU begin");
    Config config(GlobalConfig::get_config_path("config_dynamic_object_rejection"));
    mean_difference_threshold = config.param<double>("dynamic_object_rejection", "mean_difference_threshold", 0.5);
    covariance_error_threshold = config.param<double>("dynamic_object_rejection", "covariance_error_threshold", 0.5);
    points_number_difference_threshold = config.param<int>("dynamic_object_rejection", "points_number_difference_threshold", 10);
    mahalanobis_distance_threshold = config.param<double>("dynamic_object_rejection", "mahalanobis_distance_threshold", 5.0);
    dynamic_score_threshold = config.param<double>("dynamic_object_rejection", "dynamic_score_threshold", 2.5);
    num_threads = config.param<int>("dynamic_object_rejection", "num_threads", 4);
    w_shift = config.param<double>("dynamic_object_rejection", "w_shift", 1.0);
    w_mahalanobis = config.param<double>("dynamic_object_rejection", "w_mahalanobis", 0.8);
    w_covariance_difference = config.param<double>("dynamic_object_rejection", "w_covariance_difference", 0.5);
    w_shape = config.param<double>("dynamic_object_rejection", "w_shape", 0.5);
    w_occupancy = config.param<double>("dynamic_object_rejection", "w_occupancy", 0.3);
    w_neighbor = config.param<double>("dynamic_object_rejection", "w_neighbor", 0.6);
    history_factor = config.param<double>("dynamic_object_rejection", "history_factor", 0.5);
    frame_num_memory = config.param<int>("dynamic_object_rejection", "frame_num_memory", 10);
    
    
    //this one must be the same as the one used for odometry estimation to ensure consistency in voxelization
    voxel_resolution = config.param<double>("odometry_estimation", "voxel_resolution", 0.5);
    voxelmap_levels = config.param<int>("odometry_estimation", "voxelmap_levels", 2);
    voxelmap_scaling_factor = config.param<double>("odometry_estimation", "voxelmap_scaling_factor", 0.5);
}


DynamicObjectRejectionParamsCPU::~DynamicObjectRejectionParamsCPU() {
    spdlog::debug("[dynamic_rejection] DynamicObjectRejectionParamsCPU::~DynamicObjectRejectionParamsCPU");
}

DynamicObjectRejectionCPU::DynamicObjectRejectionCPU(const DynamicObjectRejectionParamsCPU& params,
                                                     const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter)
    : params_(params), pose_kalman_filter(pose_kalman_filter) {
        dynamic_voxels_indices.clear();
        covariance_estimation.reset(new CloudCovarianceEstimation(params_.num_threads));
        if (!this->pose_kalman_filter) {
            this->pose_kalman_filter = std::make_shared<PoseKalmanFilter>();
        }
        spdlog::debug("[dynamic_rejection] DynamicObjectRejectionCPU::DynamicObjectRejectionCPU");
    }



void DynamicObjectRejectionCPU::add_odometry(
    std::vector<gtsam_points::DynamicVoxelMapCPU::Ptr>& voxelmaps,
    const Eigen::Isometry3d& T_relative) {
    spdlog::debug("[dynamic_rejection] add_odometry: voxelmaps={} t=({}, {}, {})",
                  voxelmaps.size(),
                  T_relative.translation().x(),
                  T_relative.translation().y(),
                  T_relative.translation().z());
    for (auto& voxelmap : voxelmaps) {
        auto voxelmap_point_cloud = voxelmap->all_points_data();
        spdlog::debug("[dynamic_rejection] add_odometry: voxelmap all points size={}", voxelmap_point_cloud->size());
        auto transformed_pc = gtsam_points::transform(voxelmap_point_cloud, T_relative);
        auto updated_voxelmap = std::make_shared<gtsam_points::DynamicVoxelMapCPU>(params_.voxel_resolution);
        updated_voxelmap->insert(*transformed_pc);
        const int nvox =voxelmap->gtsam_points::IncrementalVoxelMap<gtsam_points::DynamicGaussianVoxel>::num_voxels();
        for (int j = 0; j < nvox; j++) {
            auto& old_voxel = voxelmap->lookup_voxel(j);

            if (!old_voxel.is_dynamic)
                continue;

            Eigen::Vector3d transformed_mean = T_relative.rotation() * old_voxel.mean.head<3>() + T_relative.translation();

            Eigen::Vector4d transformed_mean4;
            transformed_mean4 << transformed_mean, 1.0;
            auto coord = updated_voxelmap->voxel_coord(transformed_mean4);
            int new_index = updated_voxelmap->lookup_voxel_index(coord);

            if (new_index >= 0) {
                updated_voxelmap->lookup_voxel(new_index).is_dynamic = true;
            }
        }
        spdlog::debug("[dynamic_rejection] add_odometry: updated voxelmap with odometry, num_voxels={}", nvox);
        voxelmap = updated_voxelmap;
    }
}

std::vector<int> DynamicObjectRejectionCPU::get_neighbor_voxels(
    gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap,
    const Eigen::Vector4d& mean)
{
    std::vector<int> neighbors;

    auto coord = voxelmap->voxel_coord(mean);

    for(int dx=-1; dx<=1; dx++)
    for(int dy=-1; dy<=1; dy++)
    for(int dz=-1; dz<=1; dz++)
    {
        if(dx==0 && dy==0 && dz==0)
            continue;

        auto c = coord;
        c.x() += dx;
        c.y() += dy;
        c.z() += dz;

        int idx = voxelmap->lookup_voxel_index(c);

        if(idx >= 0)
            neighbors.push_back(idx);
    }

    return neighbors;
}

PreprocessedFrame::Ptr DynamicObjectRejectionCPU::dynamic_object_rejection(const PreprocessedFrame::Ptr frame){
    spdlog::debug("[dynamic_rejection] dynamic_object_rejection begin: frame={}", static_cast<bool>(frame));
    if (!frame) {
        spdlog::warn("[dynamic_rejection] dynamic_object_rejection: null input frame");
        return nullptr;
    }

    if (last_voxelmaps.empty()) {
        spdlog::debug("[dynamic_rejection] dynamic_object_rejection: no last_voxelmap, use current voxelmap for next comparison");
    }

    std::vector<Eigen::Vector4d> frame_normals;
    std::vector<Eigen::Matrix4d> frame_covs;
    
    
    spdlog::debug("[dynamic_rejection] estimating covariance for {} points", frame->points.size());
    covariance_estimation->estimate(frame->points, frame->neighbors, frame_normals, frame_covs);
    spdlog::debug("[dynamic_rejection] covariance estimation done: normals={} covs={}", frame_normals.size(), frame_covs.size());
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
    if (last_voxelmaps.empty()) {
        spdlog::debug("[dynamic_rejection] last_voxelmaps empty, skip comparison (first frame)");
        
        // Ritorna frame senza filtro dinamico se è il primo frame
        dynamic_voxels_indices.clear();
        last_voxelmaps.push_back(voxelmap);
        return frame;
    }

    //add_odometry(last_voxelmaps, pose_kalman_filter->getDeltaPose()); // for now we skip the actual transformation for simplicity; in a real implementation we would need to transform each voxel according to the estimated pose
        
    recursive_level = 1; // reset recursive level for new frame
    dynamic_voxels_indices.clear();
    static_points.clear();
    static_intensities.clear();
    static_times.clear();
    dynamic_points.clear();
    dynamic_intensities.clear();
    dynamic_times.clear();
    auto update_voxelmap = dynamic_object_recognition(voxelmap, last_voxelmaps.back());

    
    
    // Store voxelmap for next frame comparison
    last_voxelmaps.push_back(voxelmap);
    if (last_voxelmaps.size() > params_.frame_num_memory) {
        last_voxelmaps.erase(last_voxelmaps.begin());
    }
    spdlog::debug("[dynamic_rejection] stored voxelmap for next frame");

    // If no static points were accumulated, return original frame unfiltered
    if (static_points.empty()) {
        spdlog::warn("[dynamic_rejection] no static points accumulated, returning original frame");
        if (!dynamic_points.empty()) {
            last_dynamic_frame = std::make_shared<PreprocessedFrame>();
            last_dynamic_frame->stamp = frame->stamp;
            last_dynamic_frame->scan_end_time = frame->scan_end_time;
            last_dynamic_frame->points = std::move(dynamic_points);
            last_dynamic_frame->intensities = std::move(dynamic_intensities);
            last_dynamic_frame->times = std::move(dynamic_times);
            last_dynamic_frame->k_neighbors = 0;
        } else {
            last_dynamic_frame = nullptr;
        }
        return frame;
    }

    // Create output frame directly from accumulated static vectors (no PointCloudCPU intermediate)
    PreprocessedFrame::Ptr dynamic_rejection_frame(new PreprocessedFrame);
    dynamic_rejection_frame->stamp = frame->stamp;
    dynamic_rejection_frame->scan_end_time = frame->scan_end_time;
    dynamic_rejection_frame->points = std::move(static_points);
    dynamic_rejection_frame->intensities = std::move(static_intensities);
    dynamic_rejection_frame->times = std::move(static_times);
    dynamic_rejection_frame->k_neighbors = frame->k_neighbors;
    if (frame->k_neighbors > 0) {
        dynamic_rejection_frame->neighbors = find_neighbors(
            dynamic_rejection_frame->points.data(),
            dynamic_rejection_frame->points.size(),
            frame->k_neighbors);
    }
    spdlog::debug("[dynamic_rejection] frame prepared: input_points={} output_points={} dynamic_voxels={} neighbors={}",
                  frame->points.size(),
                  dynamic_rejection_frame->points.size(),
                  dynamic_voxels_indices.size(),
                  dynamic_rejection_frame->neighbors.size());

    // Build a frame containing only the dynamic points
    if (!dynamic_points.empty()) {
        last_dynamic_frame = std::make_shared<PreprocessedFrame>();
        last_dynamic_frame->stamp = frame->stamp;
        last_dynamic_frame->scan_end_time = frame->scan_end_time;
        last_dynamic_frame->points = std::move(dynamic_points);
        last_dynamic_frame->intensities = std::move(dynamic_intensities);
        last_dynamic_frame->times = std::move(dynamic_times);
        last_dynamic_frame->k_neighbors = 0;
        spdlog::debug("[dynamic_rejection] dynamic frame: {} points", last_dynamic_frame->points.size());
    } else {
        last_dynamic_frame = nullptr;
    }

    return dynamic_rejection_frame;
}



gtsam_points::DynamicVoxelMapCPU::Ptr DynamicObjectRejectionCPU::dynamic_object_recognition(gtsam_points::DynamicVoxelMapCPU::Ptr current_voxelmap, gtsam_points::DynamicVoxelMapCPU::Ptr prev_voxelmap) {
    // Compare each voxel in the current voxelmap with the corresponding voxel in the previous voxelmap
    int nvox = static_cast<int>(
        current_voxelmap->gtsam_points::IncrementalVoxelMap<gtsam_points::DynamicGaussianVoxel>::num_voxels());
    const int prev_nvox = static_cast<int>(
        prev_voxelmap->gtsam_points::IncrementalVoxelMap<gtsam_points::DynamicGaussianVoxel>::num_voxels());
    const int compare_nvox = nvox;


    for (int j = 0; j < compare_nvox; j++) {
        auto& current_voxel = current_voxelmap->lookup_voxel(j);
        // auto& prev_voxel = prev_voxelmap->lookup_voxel(j);
        auto coord = current_voxelmap->voxel_coord(current_voxel.mean);
        int voxel_index = prev_voxelmap->lookup_voxel_index(coord);
        
        if(voxel_index < 0)
        {
            // Voxel doesn't exist in previous frame → new part of environment → treat as static
            current_voxel.is_dynamic = false;
            static_points.insert(static_points.end(), current_voxel.voxel_points.begin(), current_voxel.voxel_points.end());
            static_intensities.insert(static_intensities.end(), current_voxel.voxel_intensities.begin(), current_voxel.voxel_intensities.end());
            static_times.insert(static_times.end(), current_voxel.voxel_times.begin(), current_voxel.voxel_times.end());
            continue;
        }
        auto& prev_voxel = prev_voxelmap->lookup_voxel(voxel_index);

        current_voxel.is_dynamic = false; // reset dynamic flag, will be updated by checks below
        if(prev_voxel.is_dynamic)
            current_voxel.dynamic_score=prev_voxel.dynamic_score*exp(-1.0);
        else 
            current_voxel.dynamic_score=0;
        if(current_voxel.num_points < 20 || prev_voxel.num_points < 20)
        {
            // Too few points to reliably compare → treat as static
            spdlog::debug("[dynamic_rejection] voxel {} too few points (curr={} prev={}), treating as static",j, current_voxel.num_points, prev_voxel.num_points);
            static_points.insert(static_points.end(), current_voxel.voxel_points.begin(), current_voxel.voxel_points.end());
            static_intensities.insert(static_intensities.end(), current_voxel.voxel_intensities.begin(), current_voxel.voxel_intensities.end());
            static_times.insert(static_times.end(), current_voxel.voxel_times.begin(), current_voxel.voxel_times.end());
            continue;
        }

        Eigen::Vector3d delta = (current_voxel.mean - prev_voxel.mean).head<3>();
        double mean_dist = delta.norm();

        /* ---------------------------
           1. centroid shift
           --------------------------- */

        double centroid_shift = mean_dist / current_voxelmap->voxel_resolution(); // normalize by voxel size to get a more scale-invariant measure

        spdlog::debug("[dynamic_rejection] voxel {} centroid_shift={}", j, centroid_shift);

        /* ---------------------------
           2. Mahalanobis distance
           --------------------------- */

        Eigen::Matrix3d cov =current_voxel.cov.topLeftCorner<3,3>() + prev_voxel.cov.topLeftCorner<3,3>();

        cov = 0.5 * (cov + cov.transpose());
        cov.diagonal().array() += 1e-6;

        double mahal = 0.0;

        Eigen::LDLT<Eigen::Matrix3d> ldlt(cov);

        if(ldlt.info() == Eigen::Success && ldlt.isPositive())
        {
            Eigen::Vector3d sol = ldlt.solve(delta);

            if(sol.allFinite())
            {
                double quad = delta.dot(sol);

                if(std::isfinite(quad) && quad >= 0)
                    mahal = std::sqrt(quad);
            }
        }

        spdlog::debug("[dynamic_rejection] voxel {} mahalanobis={}", j, mahal);

        /* ---------------------------
           3. covariance difference
           --------------------------- */

        double cov_norm =
            (current_voxel.cov - prev_voxel.cov).norm() /
            (current_voxel.cov.norm() + prev_voxel.cov.norm() + 1e-6);

        spdlog::debug("[dynamic_rejection] voxel {} covariance_change={}", j, cov_norm);

        /* ---------------------------
           4. shape change
           --------------------------- */

        double shape_change = 0.0;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_prev(prev_voxel.cov.topLeftCorner<3,3>());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_curr(current_voxel.cov.topLeftCorner<3,3>());

        if(eig_prev.info() == Eigen::Success &&
           eig_curr.info() == Eigen::Success)
        {
            Eigen::Vector3d l_prev = eig_prev.eigenvalues();
            Eigen::Vector3d l_curr = eig_curr.eigenvalues();
            shape_change =(l_prev - l_curr).norm() / (l_prev.norm() + 1e-6);
        }

        spdlog::debug("[dynamic_rejection] voxel {} shape_change={}", j, shape_change);

        /* ---------------------------
           5. occupancy ratio
           --------------------------- */

        double occ_ratio = std::abs((double)current_voxel.num_points - (double)prev_voxel.num_points) / (current_voxel.num_points + prev_voxel.num_points + 1e-6);

        spdlog::debug("[dynamic_rejection] voxel {} occupancy_ratio={}", j, occ_ratio);

        /* ---------------------------
           6. dynamic score
           --------------------------- */


        current_voxel.dynamic_score += params_.w_shift * centroid_shift;
        current_voxel.dynamic_score += params_.w_mahalanobis * mahal;
        current_voxel.dynamic_score += params_.w_covariance_difference * cov_norm;
        current_voxel.dynamic_score += params_.w_shape * shape_change;
        current_voxel.dynamic_score += params_.w_occupancy * occ_ratio;
        
        /* Add history factor to boost score if voxel was previously classified as dynamic */
        for (int k=0; k < params_.frame_num_memory; k++){
            if(last_voxelmaps.size() > static_cast<size_t>(k))
            {
                auto& past_map = last_voxelmaps[last_voxelmaps.size() - 1 - k];
                int past_voxel_index = past_map->lookup_voxel_index(coord);
                if(past_voxel_index < 0)
                    continue;
                auto& past_voxel = past_map->lookup_voxel(past_voxel_index);
                spdlog::debug("[dynamic_rejection] voxel {} history k={} is_dynamic={}", j, k, past_voxel.is_dynamic);
                if(past_voxel.is_dynamic)
                {
                    //current_voxel.dynamic_score += std::exp(-k) * params_.history_factor; // decaying boost for recent history
                }
                else
                {
                    current_voxel.dynamic_score -= std::exp(-k) * params_.history_factor; // decaying penalty if it was static in the past
                }
            }
        }
        
        // current_voxel.dynamic_score += params_.history_factor * (prev_voxel.is_dynamic ? 1.0 : -1.0);

        spdlog::info("[dynamic_rejection] voxel {} score={} (shift={} mahal={} cov={} shape={} occ={})",j, current_voxel.dynamic_score, centroid_shift, mahal, cov_norm, shape_change, occ_ratio);

        if(current_voxel.dynamic_score > params_.dynamic_score_threshold)
        {
            current_voxel.is_dynamic = true;
            
            spdlog::info("[dynamic_rejection] voxel {} classified as DYNAMIC", j);
        }
        else
        {
            spdlog::info("[dynamic_rejection] voxel {} classified as STATIC", j);
        }

        for (int neighbor_idx : get_neighbor_voxels(prev_voxelmap, current_voxel.mean)) {
            auto& neighbor_voxel = prev_voxelmap->lookup_voxel(neighbor_idx);
            if(current_voxel.is_dynamic)
            {
                neighbor_voxel.dynamic_score += params_.w_neighbor;
                spdlog::debug("[dynamic_rejection] voxel {} neighbor voxel {} is dynamic, increasing score to {}", j, neighbor_idx, current_voxel.dynamic_score);
            }
            else
            {
                neighbor_voxel.dynamic_score -= params_.w_neighbor;
                spdlog::debug("[dynamic_rejection] voxel {} neighbor voxel {} is static, decreasing score to {}", j, neighbor_idx, current_voxel.dynamic_score);
            }
        }
        
        // if(recursive_level < params_.voxelmap_levels && current_voxel.is_dynamic) {
        //     current_voxel.finest_voxelmap=std::make_shared<gtsam_points::DynamicVoxelMapCPU>(current_voxelmap->voxel_resolution()*params_.voxelmap_scaling_factor);
        //     current_voxel.finest_voxelmap->insert(*current_voxel.voxel_point_cloud);

        //     prev_voxel.finest_voxelmap=std::make_shared<gtsam_points::DynamicVoxelMapCPU>(current_voxelmap->voxel_resolution()*params_.voxelmap_scaling_factor);
        //     prev_voxel.finest_voxelmap->insert(*prev_voxel.voxel_point_cloud);
        //     recursive_level++;
        //     dynamic_object_recognition(current_voxel.finest_voxelmap, prev_voxel.finest_voxelmap);
        // }  
        if(!current_voxel.is_dynamic){
            // accumulate static points for output frame
            static_points.insert(static_points.end(), current_voxel.voxel_points.begin(), current_voxel.voxel_points.end());
            static_intensities.insert(static_intensities.end(), current_voxel.voxel_intensities.begin(), current_voxel.voxel_intensities.end());
            static_times.insert(static_times.end(), current_voxel.voxel_times.begin(), current_voxel.voxel_times.end());
        } else {
            // accumulate dynamic points for separate output
            dynamic_points.insert(dynamic_points.end(), current_voxel.voxel_points.begin(), current_voxel.voxel_points.end());
            dynamic_intensities.insert(dynamic_intensities.end(), current_voxel.voxel_intensities.begin(), current_voxel.voxel_intensities.end());
            dynamic_times.insert(dynamic_times.end(), current_voxel.voxel_times.begin(), current_voxel.voxel_times.end());
        }

        
        recursive_level = 1; // reset recursive level for next voxel
        
    }
    spdlog::debug("[dynamic_rejection] dynamic_object_recognition end");
    return current_voxelmap;
}



std::vector<int> DynamicObjectRejectionCPU::find_neighbors(const Eigen::Vector4d* points, const int num_points, const int k) const {
    spdlog::debug("[dynamic_rejection] find_neighbors begin: points_ptr={} num_points={} k={}", static_cast<const void*>(points), num_points, k);
    if (!points || num_points <= 0 || k <= 0) {
        spdlog::debug("[dynamic_rejection] find_neighbors bypass: invalid input");
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

    spdlog::debug("[dynamic_rejection] find_neighbors end: neighbors={}", neighbors.size());
    return neighbors;
}

}

