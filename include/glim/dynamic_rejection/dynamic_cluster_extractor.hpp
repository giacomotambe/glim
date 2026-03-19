#pragma once

#include <vector>
#include <queue>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>

#include <glim/dynamic_rejection/bounding_box.hpp>

namespace glim {

class DynamicClusterExtractor {
public:
    using VoxelMapPtr = gtsam_points::DynamicVoxelMapCPU::Ptr;

    DynamicClusterExtractor() = default;
    ~DynamicClusterExtractor() = default;

    // 1. Cluster voxel dinamici
    std::vector<std::vector<int>> cluster_voxels(VoxelMapPtr voxelmap);

    // 2. Convert voxel cluster → point cluster
    std::vector<std::vector<Eigen::Vector4d>> build_point_clusters(
        VoxelMapPtr voxelmap,
        const std::vector<std::vector<int>>& voxel_clusters);

    // 3. Bounding boxes (OBB)
    std::vector<BoundingBox> compute_bounding_boxes(
        const std::vector<std::vector<Eigen::Vector4d>>& clusters,
        int min_points = 20);

private:
    // helper
    std::vector<int> get_neighbor_voxels(
        VoxelMapPtr voxelmap,
        const Eigen::Vector4d& mean) const;

    BoundingBox createOBB(
        const std::vector<Eigen::Vector4d>& cluster) const;
};

} // namespace glim