#include "glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp"
#include <memory>
#include <fstream>
#include <iostream>
#include <unordered_set>

#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/util/fast_floor.hpp>
#include <gtsam_points/types/gaussian_voxel_data.hpp>
#include <gtsam_points/ann/incremental_voxelmap.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>

namespace gtsam_points {
template class IncrementalVoxelMap<DynamicGaussianVoxel>;


DynamicVoxelMapCPU::DynamicVoxelMapCPU(double resolution) : IncrementalVoxelMap<DynamicGaussianVoxel>(resolution) {
  offsets = neighbor_offsets(1); 
}

void DynamicGaussianVoxel::add(const GaussianVoxel::Setting& setting, const PointCloud& points, size_t i) {
    if (finalized) {
        this->finalized = false;
        this->mean *= num_points;
        this->cov *= num_points;
    }
    num_points++;
    if(points.points) {
        this->mean += points.points[i];
    }
    if(points.covs) {        
        this->cov += points.covs[i];
    }
    this->voxel_points.push_back(points.points[i]);
    if (points.intensities) {
        this->voxel_intensities.push_back(points.intensities[i]);
    }
    
    if (points.times) {
        this->voxel_times.push_back(points.times[i]);
    }

    if (frame::has_intensities(points)) {
        this->intensity = std::max(this->intensity, frame::intensity(points, i));
    }
}

DynamicGaussianVoxel& DynamicVoxelMapCPU::lookup_voxel(int voxel_id) {
    return flat_voxels[voxel_id]->second;
}

const DynamicGaussianVoxel& DynamicVoxelMapCPU::lookup_voxel(int voxel_id) const {
    return flat_voxels[voxel_id]->second;
}

void DynamicVoxelMapCPU::insert(const PointCloud& frame) {
    spdlog::debug("[DynamicVoxelMapCPU::insert] Inserting frame with {} points", frame.size());
    
    if (frame.size() == 0) {
        spdlog::warn("[DynamicVoxelMapCPU::insert] Frame has 0 points, skipping insert");
        return;
    }
    
    IncrementalVoxelMap<DynamicGaussianVoxel>::insert(frame);
    
    spdlog::debug("[DynamicVoxelMapCPU::insert] Frame inserted, initializing {} voxels", num_voxels());
    
    for (int i = 0; i < num_voxels(); i++) {
        auto& voxel = flat_voxels[i]->second;
        voxel.is_dynamic = false; // initialize as static, will be updated later in the dynamic object rejection process
        voxel.dynamic_score = 0.0; // initialize dynamic score
        voxel.voxel_point_cloud = std::make_shared<PointCloudCPU>();
        
        if (!voxel.voxel_points.empty()) {
            voxel.voxel_point_cloud->add_points(voxel.voxel_points);
        }
        
        if (!voxel.voxel_intensities.empty()) {
            voxel.voxel_point_cloud->add_intensities(voxel.voxel_intensities);
        }
        
        if (!voxel.voxel_times.empty()) {
            voxel.voxel_point_cloud->add_times(voxel.voxel_times);
        }
    
    }
    
    spdlog::debug("[DynamicVoxelMapCPU::insert] All voxels initialized");
}


PointCloudCPU::Ptr DynamicVoxelMapCPU::all_points_data() const {
    auto frame = std::make_shared<PointCloudCPU>();

    // Count total points for reservation
    size_t total = 0;
    for (int i = 0; i < num_voxels(); i++) {
        total += flat_voxels[i]->second.voxel_points.size();
    }

    frame->points_storage.reserve(total);
    std::vector<double> intensities_tmp;
    std::vector<double> times_tmp;
    std::vector<Eigen::Matrix4d> covs_tmp;
    intensities_tmp.reserve(total);
    times_tmp.reserve(total);
    covs_tmp.reserve(total);

    for (int i = 0; i < num_voxels(); i++) {
        const auto& voxel = flat_voxels[i]->second;
        const Eigen::Matrix4d& voxel_cov = voxel.cov;

        for (size_t p = 0; p < voxel.voxel_points.size(); p++) {
            frame->points_storage.emplace_back(voxel.voxel_points[p]);
            covs_tmp.emplace_back(voxel_cov);
            if (p < voxel.voxel_intensities.size()) {
                intensities_tmp.push_back(voxel.voxel_intensities[p]);
            }
            if (p < voxel.voxel_times.size()) {
                times_tmp.push_back(voxel.voxel_times[p]);
            }
        }
    }

    frame->num_points = frame->points_storage.size();
    frame->points = frame->points_storage.data();

    if (!intensities_tmp.empty()) {
        frame->add_intensities(intensities_tmp);
    }
    if (!times_tmp.empty()) {
        frame->add_times(times_tmp);
    }
    if (!covs_tmp.empty()) {
        frame->add_covs(covs_tmp);
    }

    return frame;
}

double DynamicVoxelMapCPU::voxel_resolution() const {
    return leaf_size();
}


Eigen::Vector3i DynamicVoxelMapCPU::voxel_coord(const Eigen::Vector4d& x) const {
    return fast_floor(x * inv_leaf_size).head<3>();
}

int DynamicVoxelMapCPU::lookup_voxel_index(const Eigen::Vector3i& coord) const {
    auto found = voxels.find(coord);
    if (found == voxels.end()) {
        return -1;
    }
    return found->second;

}

void DynamicVoxelMapCPU::save_compact(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "error: failed to open " << path << " for writing" << std::endl;
        return;
    }

    // Write header
    ofs << "dynamic_compact " << 1 << std::endl;
    ofs << "resolution " << voxel_resolution() << std::endl;
    ofs << "num_voxels " << flat_voxels.size() << std::endl;

    // Write voxel data
    for (const auto& voxel_pair : flat_voxels) {
        const DynamicGaussianVoxel& voxel = voxel_pair->second;
        
        // Write voxel coordinate
        const Eigen::Vector3i& coord = voxel_pair->first.coord;
        ofs.write(reinterpret_cast<const char*>(&coord.x()), sizeof(int) * 3);
        
        // Write voxel core data (Gaussian statistics)
        ofs.write(reinterpret_cast<const char*>(&voxel.num_points), sizeof(voxel.num_points));
        ofs.write(reinterpret_cast<const char*>(&voxel.mean.x()), sizeof(double) * 4);
        ofs.write(reinterpret_cast<const char*>(&voxel.cov.data()[0]), sizeof(double) * 16);
        
        // Write dynamic-specific data
        ofs.write(reinterpret_cast<const char*>(&voxel.is_dynamic), sizeof(voxel.is_dynamic));
        
        size_t num_points_data = voxel.voxel_points.size();
        ofs.write(reinterpret_cast<const char*>(&num_points_data), sizeof(num_points_data));
        
        for (const auto& pt : voxel.voxel_points) {
            ofs.write(reinterpret_cast<const char*>(&pt.x()), sizeof(double) * 4);
        }
        
        for (const auto& intensity : voxel.voxel_intensities) {
            ofs.write(reinterpret_cast<const char*>(&intensity), sizeof(double));
        }
        
        for (const auto& time : voxel.voxel_times) {
            ofs.write(reinterpret_cast<const char*>(&time), sizeof(double));
        }
    }
}


}