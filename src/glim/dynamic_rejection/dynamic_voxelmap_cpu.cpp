#include "glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp"

namespace dynamic_glim {
template class IncrementalVoxelMap<DynamicGaussianVoxel>;

void DynamicGaussianVoxel::add(const Setting& setting, const PointCloud& points, size_t i) {
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

    if (frame::has_intensities(points)) {
        this->intensity = std::max(this->intensity, frame::intensity(points, i));
    }
}

const DynamicGaussianVoxel& DynamicVoxelMapCPU::lookup_voxel(int voxel_id) const {
    return flat_voxels[voxel_id]->second;
}

void DynamicVoxelMapCPU::insert(const PointCloud& frame) {
    IncrementalVoxelMap<DynamicGaussianVoxel>::insert(frame);
    // After inserting the frame, we need to update the is_dynamic flag of each voxel based on the new data. This is a placeholder for the actual logic to determine if a voxel is dynamic or not, which
    // would likely involve comparing the new voxel data with previous frames or using some heuristic based on the voxel's properties (e.g., mean, covariance, number of points).
    for (int i = 0; i < num_voxel(); i++) {
        auto& voxel = flat_voxels[i]->second;
        voxel.is_dynamic = true;
    }
}
}