#include <glim/dynamic_rejection/dynamic_cluster_extractor.hpp>

#include <algorithm>
#include <spdlog/spdlog.h>
#include <gtsam_points/ann/kdtree.hpp>
#include <glim/util/config.hpp>

namespace glim {

// ===========================================================================
// DynamicClusterExtractorParams
// ===========================================================================

DynamicClusterExtractorParams::DynamicClusterExtractorParams() {
    spdlog::debug("[cluster_extractor] loading config");

    Config config(GlobalConfig::get_config_path("config_dynamic_cluster_extractor"));

    eps_voxel_factor    = config.param<double>("dynamic_cluster_extractor", "eps_voxel_factor",    2.0);
    min_pts             = config.param<int>   ("dynamic_cluster_extractor", "min_pts",             1);
    knn_max_neighbors   = config.param<int>   ("dynamic_cluster_extractor", "knn_max_neighbors",   64);
    min_cluster_voxels  = config.param<int>   ("dynamic_cluster_extractor", "min_cluster_voxels",  2);
    min_points_for_bbox = config.param<int>   ("dynamic_cluster_extractor", "min_points_for_bbox", 20);

    // Filtri dimensionali — tutti opzionali (default = nessun filtro)
    bbox_min_extent = config.param<double>("dynamic_cluster_extractor", "bbox_min_extent", 0.0);
    bbox_max_extent = config.param<double>("dynamic_cluster_extractor", "bbox_max_extent", 1e9);
    bbox_min_volume = config.param<double>("dynamic_cluster_extractor", "bbox_min_volume", 0.0);
    bbox_max_volume = config.param<double>("dynamic_cluster_extractor", "bbox_max_volume", 1e9);

    spdlog::debug("[cluster_extractor] eps_factor={:.2f} min_pts={} knn_max={} "
                  "min_cluster_voxels={} min_points_bbox={} "
                  "bbox_extent=[{:.2f},{:.2f}] bbox_volume=[{:.3f},{:.3f}]",
                  eps_voxel_factor, min_pts, knn_max_neighbors,
                  min_cluster_voxels, min_points_for_bbox,
                  bbox_min_extent, bbox_max_extent,
                  bbox_min_volume, bbox_max_volume);
}

DynamicClusterExtractorParams::~DynamicClusterExtractorParams() = default;

// ===========================================================================
// Costruttori
// ===========================================================================

DynamicClusterExtractor::DynamicClusterExtractor()
    : params_() {}

DynamicClusterExtractor::DynamicClusterExtractor(
    const DynamicClusterExtractorParams& params)
    : params_(params) {}

// ===========================================================================
// extract_clusters()
// ===========================================================================

std::vector<BoundingBox> DynamicClusterExtractor::extract_clusters(
    gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap) const
{
    const auto cluster_map = cluster_voxels(voxelmap);

    int num_clusters = 0;
    for (int id : cluster_map) {
        if (id >= num_clusters) num_clusters = id + 1;
    }
    spdlog::debug("[cluster_extractor] found {} clusters", num_clusters);
    const auto clusters = build_point_clusters(voxelmap, cluster_map, num_clusters);
    return compute_bounding_boxes(clusters);
}

// ===========================================================================
// cluster_voxels()  —  DBSCAN sui centroidi dei voxel NON-wall
// ===========================================================================

DynamicClusterExtractor::ClusterMap
DynamicClusterExtractor::cluster_voxels(gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap) const
{
    const int nvox = static_cast<int>(
        voxelmap->gtsam_points::IncrementalVoxelMap<
            gtsam_points::DynamicGaussianVoxel>::num_voxels());

    ClusterMap cluster_map(nvox, -1);

    std::vector<int>             active_ids;
    std::vector<Eigen::Vector4d> active_cents;
    active_ids.reserve(nvox);
    active_cents.reserve(nvox);

    for (int i = 0; i < nvox; ++i) {
        const auto& v = voxelmap->lookup_voxel(i);
        if (!v.is_wall) {
            active_ids.push_back(i);
            active_cents.push_back(v.mean);
        }
    }

    const int n_active = static_cast<int>(active_ids.size());
    spdlog::debug("[DBSCAN] nvox={} active(non-wall)={}", nvox, n_active);
    if (n_active == 0) return cluster_map;

    const double voxel_res = voxelmap->voxel_resolution();
    const double eps       = params_.eps_voxel_factor * voxel_res;
    const double eps2      = eps * eps;
    const int    min_pts   = params_.min_pts;
    const int    knn_k     = std::min(n_active, params_.knn_max_neighbors);

    spdlog::debug("[DBSCAN] voxel_res={:.3f} eps={:.3f} min_pts={} knn_k={}",
                  voxel_res, eps, min_pts, knn_k);

    gtsam_points::KdTree tree(active_cents.data(), n_active);

    auto range_query = [&](int local_idx) -> std::vector<int> {
        std::vector<size_t> k_idx(knn_k);
        std::vector<double> k_sq (knn_k);
        const size_t found = tree.knn_search(
            active_cents[local_idx].data(), knn_k, k_idx.data(), k_sq.data());

        std::vector<int> neighbors;
        neighbors.reserve(found);
        for (size_t i = 0; i < found; ++i) {
            if (k_sq[i] <= eps2 && static_cast<int>(k_idx[i]) != local_idx)
                neighbors.push_back(static_cast<int>(k_idx[i]));
        }
        return neighbors;
    };

    constexpr int UNVISITED = -2;
    constexpr int NOISE     = -1;

    std::vector<int> label(n_active, UNVISITED);
    int cluster_id = 0;

    for (int i = 0; i < n_active; ++i) {
        if (label[i] != UNVISITED) continue;

        const auto neighbors = range_query(i);
        if (static_cast<int>(neighbors.size()) < min_pts) {
            label[i] = NOISE;
            continue;
        }

        label[i] = cluster_id;
        std::vector<int> seed_set = neighbors;

        for (int s = 0; s < static_cast<int>(seed_set.size()); ++s) {
            const int q = seed_set[s];
            if (label[q] == NOISE) { label[q] = cluster_id; continue; }
            if (label[q] != UNVISITED) continue;

            label[q] = cluster_id;
            const auto q_nbrs = range_query(q);
            if (static_cast<int>(q_nbrs.size()) >= min_pts) {
                for (int nb : q_nbrs) {
                    if (label[nb] == UNVISITED || label[nb] == NOISE)
                        seed_set.push_back(nb);
                }
            }
        }
        ++cluster_id;
    }

    spdlog::debug("[DBSCAN] raw_clusters={}", cluster_id);

    // Filtra cluster troppo piccoli e rinumera
    std::vector<int> cluster_size(cluster_id, 0);
    for (int i = 0; i < n_active; ++i)
        if (label[i] >= 0) ++cluster_size[label[i]];

    std::vector<int> remap(cluster_id, -1);
    int valid_id = 0;
    for (int c = 0; c < cluster_id; ++c)
        if (cluster_size[c] >= params_.min_cluster_voxels)
            remap[c] = valid_id++;

    spdlog::debug("[DBSCAN] {} clusters after size filter (min_voxels={})",
                  valid_id, params_.min_cluster_voxels);

    for (int i = 0; i < n_active; ++i)
        if (label[i] >= 0 && remap[label[i]] >= 0)
            cluster_map[active_ids[i]] = remap[label[i]];

    return cluster_map;
}

// ===========================================================================
// build_point_clusters()
// ===========================================================================

std::vector<std::vector<Eigen::Vector4d>>
DynamicClusterExtractor::build_point_clusters(
    VoxelMapPtr       voxelmap,
    const ClusterMap& cluster_map,
    int               num_clusters) const
{
    std::vector<std::vector<Eigen::Vector4d>> point_clusters(num_clusters);

    const int nvox = static_cast<int>(cluster_map.size());
    for (int i = 0; i < nvox; ++i) {
        const int cid = cluster_map[i];
        if (cid < 0) continue;
        const auto& voxel = voxelmap->lookup_voxel(i);
        point_clusters[cid].insert(
            point_clusters[cid].end(),
            voxel.voxel_points.begin(),
            voxel.voxel_points.end());
    }

    return point_clusters;
}

// ===========================================================================
// compute_bounding_boxes()
// ===========================================================================

std::vector<BoundingBox>
DynamicClusterExtractor::compute_bounding_boxes(
    const std::vector<std::vector<Eigen::Vector4d>>& clusters) const
{
    return compute_bounding_boxes(clusters, params_.min_points_for_bbox);
}

std::vector<BoundingBox>
DynamicClusterExtractor::compute_bounding_boxes(
    const std::vector<std::vector<Eigen::Vector4d>>& clusters,
    int min_points) const
{
    std::vector<BoundingBox> boxes;
    boxes.reserve(clusters.size());

    int rejected_pts     = 0;
    int rejected_extent  = 0;
    int rejected_volume  = 0;

    for (const auto& cluster : clusters) {
        // --- Filtro numero minimo di punti ---
        if (static_cast<int>(cluster.size()) < min_points) {
            ++rejected_pts;
            continue;
        }

        // --- Crea OBB e applica i filtri dimensionali ---
        BoundingBox bbox(Eigen::Vector3d::Zero(),
                         Eigen::Vector3d::Zero(),
                         Eigen::Matrix3d::Identity());
        if (!createOBB(cluster, bbox)) {
            // Il tipo di rifiuto è già loggato dentro createOBB
            // Contiamo per il debug qui sotto
            ++rejected_extent;   // ricicliamo questo contatore per tutti i filtri geometrici
            continue;
        }

        boxes.push_back(bbox);
    }

    spdlog::debug("[cluster_extractor] {} bounding boxes kept "
                  "(rejected: {} pts, {} size/vol, min_pts={})",
                  boxes.size(), rejected_pts, rejected_extent, min_points);
    return boxes;
}

// ===========================================================================
// createOBB()  —  costruisce l'OBB e applica i filtri dimensionali
//
// Ritorna false se l'OBB non supera i filtri; in quel caso out_bbox non
// viene modificata.
// ===========================================================================

bool DynamicClusterExtractor::createOBB(
    const std::vector<Eigen::Vector4d>& cluster,
    BoundingBox& out_bbox) const
{
    // --- Centroide ---
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& p : cluster) mean += p.head<3>();
    mean /= static_cast<double>(cluster.size());

    // --- Matrice di covarianza ---
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : cluster) {
        const Eigen::Vector3d d = p.head<3>() - mean;
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(cluster.size());

    // --- Assi principali (PCA) ---
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    const Eigen::Matrix3d R = solver.eigenvectors();

    // --- AABB nel frame locale ---
    Eigen::Vector3d local_min( 1e9,  1e9,  1e9);
    Eigen::Vector3d local_max(-1e9, -1e9, -1e9);
    for (const auto& p : cluster) {
        const Eigen::Vector3d local = R.transpose() * (p.head<3>() - mean);
        local_min = local_min.cwiseMin(local);
        local_max = local_max.cwiseMax(local);
    }

    const Eigen::Vector3d size         = local_max - local_min;
    const Eigen::Vector3d center_local = 0.5 * (local_max + local_min);
    const Eigen::Vector3d center       = R * center_local + mean;

    // -----------------------------------------------------------------------
    // Filtri dimensionali
    // -----------------------------------------------------------------------
    const double min_dim = size.minCoeff();
    const double max_dim = size.maxCoeff();
    const double volume  = size.x() * size.y() * size.z();

    // 1. Lato minimo troppo piccolo (oggetto quasi planare / rumore)
    if (params_.bbox_min_extent > 0.0 && min_dim < params_.bbox_min_extent) {
        spdlog::debug("[cluster_extractor] bbox rejected: min_extent {:.3f} < {:.3f}",
                      min_dim, params_.bbox_min_extent);
        return false;
    }

    // 2. Lato massimo troppo grande (falso positivo, muro, etc.)
    if (params_.bbox_max_extent < 1e8 && max_dim > params_.bbox_max_extent) {
        spdlog::debug("[cluster_extractor] bbox rejected: max_extent {:.3f} > {:.3f}",
                      max_dim, params_.bbox_max_extent);
        return false;
    }

    // 3. Volume troppo piccolo
    if (params_.bbox_min_volume > 0.0 && volume < params_.bbox_min_volume) {
        spdlog::debug("[cluster_extractor] bbox rejected: volume {:.4f} < {:.4f}",
                      volume, params_.bbox_min_volume);
        return false;
    }

    // 4. Volume troppo grande
    if (params_.bbox_max_volume < 1e8 && volume > params_.bbox_max_volume) {
        spdlog::debug("[cluster_extractor] bbox rejected: volume {:.4f} > {:.4f}",
                      volume, params_.bbox_max_volume);
        return false;
    }

    out_bbox = BoundingBox(size, center, R);
    return true;
}

} // namespace glim