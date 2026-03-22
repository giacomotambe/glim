#pragma once

#include <vector>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp>
#include <glim/dynamic_rejection/bounding_box.hpp>

namespace glim {

// ===========================================================================
// DynamicClusterExtractorParams
//
// Stesso pattern di DynamicObjectRejectionParamsCPU: il costruttore legge
// dal file "config_dynamic_cluster_extractor" tramite glim::Config.
// ===========================================================================
struct DynamicClusterExtractorParams {
public:
    DynamicClusterExtractorParams();
    ~DynamicClusterExtractorParams();

public:
    // --- DBSCAN ---

    /// Moltiplicatore applicato alla voxel_resolution per ottenere eps.
    /// eps = eps_voxel_factor * voxel_resolution
    /// Default: 2.0
    double eps_voxel_factor;

    /// Numero minimo di vicini (escluso se stesso) per essere core-point.
    /// Default: 1
    int min_pts;

    /// Numero massimo di vicini cercati nella knn_search della range query.
    /// Default: 64
    int knn_max_neighbors;

    // --- Filtro dimensione cluster ---

    /// Numero minimo di voxel per tenere un cluster.
    /// Default: 2
    int min_cluster_voxels;

    /// Numero minimo di punti raw per generare una bounding box.
    /// Default: 20
    int min_points_for_bbox;
};

// ===========================================================================
// DynamicClusterExtractor
//
// Uso nella pipeline (chiamato PRIMA di DynamicObjectRejectionCPU::reject()):
//
//   WallFilterResult wf       = wall_filter.filter(frame);
//   ClusterMap       cmap     = extractor.cluster_voxels(wf.voxelmap);
//   DynamicRejectionResult dr = rejector.reject(wf, frame, cmap);
//
// cluster_map[voxel_id]:
//   >= 0  → ID del cluster a cui appartiene il voxel
//   == -1 → voxel wall oppure NOISE DBSCAN (non appartiene a nessun cluster)
// ===========================================================================
class DynamicClusterExtractor {
public:
    using VoxelMapPtr = gtsam_points::DynamicVoxelMapCPU::Ptr;

    /// cluster_map[voxel_id] = cluster_id  (-1 → wall / NOISE)
    using ClusterMap  = std::vector<int>;

    DynamicClusterExtractor();
    explicit DynamicClusterExtractor(const DynamicClusterExtractorParams& params);
    ~DynamicClusterExtractor() = default;

    /**
     * @brief  DBSCAN sui centroidi dei voxel NON-wall.
     *
     * Salta i voxel con is_wall == true (cluster_map[i] = -1 per loro).
     * Restituisce un vettore di dimensione num_voxels():
     *   cluster_map[i] >= 0  → cluster ID
     *   cluster_map[i] == -1 → wall o NOISE
     *
     * @param voxelmap  Output di WallFilter::filter() con is_wall marcato.
     * @return          ClusterMap indicizzata per voxel_id globale.
     */
    ClusterMap cluster_voxels(gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap) const;

    // -----------------------------------------------------------------------
    // Utilities post-rejection (per visualizzazione / bbox)
    // -----------------------------------------------------------------------

    /// Raccoglie i punti raw di ogni cluster dalla voxelmap.
    /// num_clusters = valore massimo in cluster_map + 1.
    std::vector<std::vector<Eigen::Vector4d>> build_point_clusters(
        VoxelMapPtr       voxelmap,
        const ClusterMap& cluster_map,
        int               num_clusters) const;

    /// Genera OBB per ogni cluster con abbastanza punti (usa params_.min_points_for_bbox).
    std::vector<BoundingBox> compute_bounding_boxes(
        const std::vector<std::vector<Eigen::Vector4d>>& clusters) const;

    /// Overload con soglia esplicita.
    std::vector<BoundingBox> compute_bounding_boxes(
        const std::vector<std::vector<Eigen::Vector4d>>& clusters,
        int min_points) const;

    const DynamicClusterExtractorParams& params() const { return params_; }

    std::vector<BoundingBox> extract_clusters(gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap) const;

private:
    BoundingBox createOBB(const std::vector<Eigen::Vector4d>& cluster) const;

private:
    DynamicClusterExtractorParams params_;
};

} // namespace glim