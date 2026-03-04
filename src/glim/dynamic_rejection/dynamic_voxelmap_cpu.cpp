#include "glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp"
#include <memory>
#include <fstream>
#include <iostream>
#include <unordered_set>

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
    this->mean += points.points[i];
    this->cov += points.covs[i];
    this->voxel_points.push_back(points.points[i]);
    this->voxel_intensities.push_back(points.intensities[i]);
    this->voxel_times.push_back(points.times[i]);
    this->is_dynamic = true; // default to dynamic until proven otherwise

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
    IncrementalVoxelMap<DynamicGaussianVoxel>::insert(frame);
    // // After inserting the frame, we need to update the is_dynamic flag of each voxel based on the new data. This is a placeholder for the actual logic to determine if a voxel is dynamic or not, which
    // // would likely involve comparing the new voxel data with previous frames or using some heuristic based on the voxel's properties (e.g., mean, covariance, number of points).
    // for (int i = 0; i < num_voxels(); i++) {
    //     auto& voxel = flat_voxels[i]->second;
    //     voxel.is_dynamic = true;
    // }
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