#include "dynamic_cluster_extractor.hpp"

namespace glim {

std::vector<std::vector<int>> DynamicClusterExtractor::cluster_voxels(VoxelMapPtr voxelmap)
{
    int nvox = static_cast<int>(
        voxelmap->gtsam_points::IncrementalVoxelMap<
        gtsam_points::DynamicGaussianVoxel>::num_voxels());

    std::vector<std::vector<int>> clusters;
    std::vector<bool> visited(nvox, false);

    for (int i = 0; i < nvox; i++) {
        auto& v = voxelmap->lookup_voxel(i);

        if (!v.is_dynamic || visited[i])
            continue;

        std::vector<int> cluster;
        std::queue<int> q;

        q.push(i);
        visited[i] = true;

        while (!q.empty()) {
            int idx = q.front();
            q.pop();

            cluster.push_back(idx);

            auto& voxel = voxelmap->lookup_voxel(idx);
            auto neighbors = get_neighbor_voxels(voxelmap, voxel.mean);

            for (auto n : neighbors) {
                if (n < 0 || n >= nvox)
                    continue;

                if (!visited[n] && voxelmap->lookup_voxel(n).is_dynamic) {
                    visited[n] = true;
                    q.push(n);
                }
            }
        }

        if (cluster.size() > 1)
            clusters.push_back(cluster);
    }

    return clusters;
}

std::vector<std::vector<Eigen::Vector4d>>
DynamicClusterExtractor::build_point_clusters(
    VoxelMapPtr voxelmap,
    const std::vector<std::vector<int>>& voxel_clusters)
{
    std::vector<std::vector<Eigen::Vector4d>> point_clusters;

    for (const auto& voxel_cluster : voxel_clusters) {
        std::vector<Eigen::Vector4d> cluster_points;

        for (auto idx : voxel_cluster) {
            auto& voxel = voxelmap->lookup_voxel(idx);

            cluster_points.insert(cluster_points.end(),
                voxel.voxel_points.begin(),
                voxel.voxel_points.end());
        }

        if (!cluster_points.empty())
            point_clusters.push_back(cluster_points);
    }

    return point_clusters;
}

std::vector<BoundingBox>
DynamicClusterExtractor::compute_bounding_boxes(
    const std::vector<std::vector<Eigen::Vector4d>>& clusters,
    int min_points)
{
    std::vector<BoundingBox> boxes;

    for (const auto& cluster : clusters) {
        if (static_cast<int>(cluster.size()) < min_points)
            continue;

        boxes.push_back(createOBB(cluster));
    }

    return boxes;
}

std::vector<int> DynamicClusterExtractor::get_neighbor_voxels(
    VoxelMapPtr voxelmap,
    const Eigen::Vector4d& mean) const
{
    std::vector<int> neighbors;

    auto coord = voxelmap->voxel_coord(mean);

    for (int dx = -1; dx <= 1; dx++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dz = -1; dz <= 1; dz++)
    {
        if (dx == 0 && dy == 0 && dz == 0)
            continue;

        auto c = coord;
        c.x() += dx;
        c.y() += dy;
        c.z() += dz;

        int idx = voxelmap->lookup_voxel_index(c);

        if (idx >= 0)
            neighbors.push_back(idx);
    }

    return neighbors;
}

BoundingBox DynamicClusterExtractor::createOBB(
    const std::vector<Eigen::Vector4d>& cluster) const
{
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();

    for (const auto& p : cluster)
        mean += p.head<3>();

    mean /= cluster.size();

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

    for (const auto& p : cluster) {
        Eigen::Vector3d d = p.head<3>() - mean;
        cov += d * d.transpose();
    }

    cov /= cluster.size();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Matrix3d R = solver.eigenvectors();

    Eigen::Vector3d min_pt(1e9,1e9,1e9);
    Eigen::Vector3d max_pt(-1e9,-1e9,-1e9);

    for (const auto& p : cluster) {
        Eigen::Vector3d local = R.transpose() * (p.head<3>() - mean);
        min_pt = min_pt.cwiseMin(local);
        max_pt = max_pt.cwiseMax(local);
    }

    Eigen::Vector3d size = max_pt - min_pt;
    Eigen::Vector3d center_local = 0.5 * (max_pt + min_pt);
    Eigen::Vector3d center = R * center_local + mean;

    return BoundingBox(size, center, R);
}

} // namespace glim