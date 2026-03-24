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

    // --- Filtro dimensionale sulle bounding box ---

    /// Dimensione minima ammessa lungo ciascun asse dell'OBB [m].
    /// Un'OBB viene scartata se il suo lato più corto è < bbox_min_extent.
    /// Utile per eliminare cluster rumorosi troppo piccoli.
    /// Default: 0.0  (nessun filtro)
    double bbox_min_extent;

    /// Dimensione massima ammessa lungo ciascun asse dell'OBB [m].
    /// Un'OBB viene scartata se il suo lato più lungo è > bbox_max_extent.
    /// Utile per eliminare falsi positivi enormi (es. muri non classificati).
    /// Default: 1e9  (nessun filtro)
    double bbox_max_extent;

    /// Volume minimo ammesso [m³].
    /// Default: 0.0  (nessun filtro)
    double bbox_min_volume;

    /// Volume massimo ammesso [m³].
    /// Default: 1e9  (nessun filtro)
    double bbox_max_volume;
};

// ===========================================================================
// DynamicClusterExtractor
// ===========================================================================
class DynamicClusterExtractor {
public:
    using VoxelMapPtr = gtsam_points::DynamicVoxelMapCPU::Ptr;
    using ClusterMap  = std::vector<int>;

    DynamicClusterExtractor();
    explicit DynamicClusterExtractor(const DynamicClusterExtractorParams& params);
    ~DynamicClusterExtractor() = default;

    /**
     * @brief  DBSCAN sui centroidi dei voxel NON-wall.
     */
    ClusterMap cluster_voxels(gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap) const;

    // -----------------------------------------------------------------------
    // Utilities post-rejection
    // -----------------------------------------------------------------------

    std::vector<std::vector<Eigen::Vector4d>> build_point_clusters(
        VoxelMapPtr       voxelmap,
        const ClusterMap& cluster_map,
        int               num_clusters) const;

    /// Genera OBB per ogni cluster con abbastanza punti, applicando i filtri
    /// dimensionali configurati (bbox_min/max_extent, bbox_min/max_volume).
    std::vector<BoundingBox> compute_bounding_boxes(
        const std::vector<std::vector<Eigen::Vector4d>>& clusters) const;

    /// Overload con soglia minima punti esplicita (usa comunque i filtri dimensionali).
    std::vector<BoundingBox> compute_bounding_boxes(
        const std::vector<std::vector<Eigen::Vector4d>>& clusters,
        int min_points) const;

    const DynamicClusterExtractorParams& params() const { return params_; }

    std::vector<BoundingBox> extract_clusters(gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap) const;

private:
    /// Crea l'OBB per un singolo cluster.  Ritorna true se l'OBB supera i
    /// filtri dimensionali configurati, false se deve essere scartata.
    bool createOBB(const std::vector<Eigen::Vector4d>& cluster,
                   BoundingBox& out_bbox) const;

private:
    DynamicClusterExtractorParams params_;
};

} // namespace glim