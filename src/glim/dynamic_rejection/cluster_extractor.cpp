/*
    FILE: cluster_extractor.cpp
    ------------------
    Improved dynamic cluster extractor.

    Key changes vs v2:
      5. merge_with_history(): new method implementing temporal bbox merging.
         If a large historical bbox contains one or more current small bboxes,
         those are replaced by the historical bbox (marked static).
         This handles the case where a single object is over-segmented into
         multiple small clusters in the current frame.
      6. classify_clusters() now calls merge_with_history() first, then runs
         the standard static/dynamic classification on the remaining bboxes.
*/

#include <glim/dynamic_rejection/cluster_extractor.hpp>

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

    bbox_min_extent = config.param<double>("dynamic_cluster_extractor", "bbox_min_extent", 0.0);
    bbox_max_extent = config.param<double>("dynamic_cluster_extractor", "bbox_max_extent", 1e9);
    bbox_min_volume = config.param<double>("dynamic_cluster_extractor", "bbox_min_volume", 0.0);
    bbox_max_volume = config.param<double>("dynamic_cluster_extractor", "bbox_max_volume", 1e9);

    cluster_distance_threshold = config.param<double>("dynamic_cluster_extractor", "cluster_distance_threshold", 0.5);
    cluster_iou_threshold      = config.param<double>("dynamic_cluster_extractor", "cluster_iou_threshold",      0.5);
    history_size               = config.param<int>   ("dynamic_cluster_extractor", "history_size",               5);
    min_static_history_matches = config.param<int>   ("dynamic_cluster_extractor", "min_static_history_matches", 3);

    // New param: a current bbox is considered "contained" in a historical one
    // if its center lies inside the historical bbox expanded by this margin (metres).
    containment_margin = config.param<double>("dynamic_cluster_extractor", "containment_margin", 0.1);

    // New param: minimum volume ratio hist/current to trigger merging.
    // Avoids replacing a large current bbox with an equally-large historical one.
    merge_volume_ratio = config.param<double>("dynamic_cluster_extractor", "merge_volume_ratio", 1.5);

    spdlog::debug("[cluster_extractor] eps_factor={:.2f} min_pts={} knn_max={} "
                  "min_cluster_voxels={} min_points_bbox={} "
                  "bbox_extent=[{:.2f},{:.2f}] bbox_volume=[{:.3f},{:.3f}] "
                  "cluster_distance_threshold={:.2f} cluster_iou_threshold={:.2f} "
                  "history_size={} min_static_history_matches={} "
                  "containment_margin={:.2f} merge_volume_ratio={:.2f}",
                  eps_voxel_factor, min_pts, knn_max_neighbors,
                  min_cluster_voxels, min_points_for_bbox,
                  bbox_min_extent, bbox_max_extent,
                  bbox_min_volume, bbox_max_volume,
                  cluster_distance_threshold, cluster_iou_threshold,
                  history_size, min_static_history_matches,
                  containment_margin, merge_volume_ratio);
}

DynamicClusterExtractorParams::~DynamicClusterExtractorParams() = default;

// ===========================================================================
// Constructors
// ===========================================================================

DynamicClusterExtractor::DynamicClusterExtractor(
    const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter)
    : params_(), pose_kalman_filter_(pose_kalman_filter) {}

DynamicClusterExtractor::DynamicClusterExtractor(
    const DynamicClusterExtractorParams& params)
    : params_(params) {}

// ===========================================================================
// NMS utility
// ===========================================================================

static std::vector<BoundingBox> nms3D(
    const std::vector<BoundingBox>& boxes,
    double iou_threshold)
{
    const int N = static_cast<int>(boxes.size());
    if (N == 0) return {};

    struct Candidate { int idx; double score; };
    std::vector<Candidate> candidates;
    candidates.reserve(N);
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d& s = boxes[i].get_size();
        candidates.push_back({i, s.x() * s.y() * s.z()});
    }
    std::sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b){ return a.score > b.score; });

    std::vector<bool> suppressed(N, false);
    std::vector<BoundingBox> result;
    for (int i = 0; i < N; ++i) {
        const int ii = candidates[i].idx;
        if (suppressed[ii]) continue;
        result.push_back(boxes[ii]);
        for (int j = i + 1; j < N; ++j) {
            const int jj = candidates[j].idx;
            if (!suppressed[jj] && boxes[ii].iou(boxes[jj]) > iou_threshold)
                suppressed[jj] = true;
        }
    }
    return result;
}

// ===========================================================================
// extract_clusters()
// ===========================================================================

std::vector<BoundingBox> DynamicClusterExtractor::extract_clusters(
    gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap)
{
    const auto cluster_map = cluster_voxels(voxelmap);

    int num_clusters = 0;
    for (int id : cluster_map)
        if (id >= num_clusters) num_clusters = id + 1;

    spdlog::debug("[cluster_extractor] found {} clusters", num_clusters);

    const auto clusters = build_point_clusters(voxelmap, cluster_map, num_clusters);
    auto bboxes = compute_bounding_boxes(clusters);
    bboxes = nms3D(bboxes, params_.cluster_iou_threshold);

    spdlog::debug("[cluster_extractor] {} bboxes after NMS", bboxes.size());

    if (pose_kalman_filter_) {
        Eigen::Isometry3d cur_pose = pose_kalman_filter_->getDeltaPose();
        Eigen::Isometry3d T_delta  = cur_pose * last_pose_.inverse();
        last_pose_ = cur_pose;
        classify_clusters(T_delta, bboxes);
    }

    return bboxes;
}

// ===========================================================================
// cluster_voxels()  —  DBSCAN on non-wall voxel centroids
// ===========================================================================

DynamicClusterExtractor::ClusterMap
DynamicClusterExtractor::cluster_voxels(
    gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap) const
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
        for (size_t i = 0; i < found; ++i)
            if (k_sq[i] <= eps2 && static_cast<int>(k_idx[i]) != local_idx)
                neighbors.push_back(static_cast<int>(k_idx[i]));
        return neighbors;
    };

    constexpr int UNVISITED = -2;
    constexpr int NOISE_LBL = -1;

    std::vector<int>  label(n_active, UNVISITED);
    std::vector<bool> in_seed(n_active, false);
    int cluster_id = 0;

    for (int i = 0; i < n_active; ++i) {
        if (label[i] != UNVISITED) continue;

        const auto neighbors = range_query(i);
        if (static_cast<int>(neighbors.size()) < min_pts) {
            label[i] = NOISE_LBL;
            continue;
        }

        label[i] = cluster_id;
        std::vector<int> seed_set = neighbors;
        for (int nb : neighbors) in_seed[nb] = true;

        for (int s = 0; s < static_cast<int>(seed_set.size()); ++s) {
            const int q = seed_set[s];
            if (label[q] == NOISE_LBL) { label[q] = cluster_id; continue; }
            if (label[q] != UNVISITED)  continue;

            label[q] = cluster_id;
            const auto q_nbrs = range_query(q);
            if (static_cast<int>(q_nbrs.size()) >= min_pts) {
                for (int nb : q_nbrs) {
                    if ((label[nb] == UNVISITED || label[nb] == NOISE_LBL) && !in_seed[nb]) {
                        seed_set.push_back(nb);
                        in_seed[nb] = true;
                    }
                }
            }
        }

        // Reset in_seed for next cluster (only touch used entries)
        for (int nb : seed_set) in_seed[nb] = false;
        in_seed[i] = false;

        ++cluster_id;
    }

    spdlog::debug("[DBSCAN] raw_clusters={}", cluster_id);

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

    int rejected_pts  = 0;
    int rejected_geom = 0;

    for (const auto& cluster : clusters) {
        if (static_cast<int>(cluster.size()) < min_points) {
            ++rejected_pts;
            continue;
        }

        BoundingBox bbox(Eigen::Vector3d::Zero(),
                         Eigen::Vector3d::Zero(),
                         Eigen::Matrix3d::Identity());
        if (!createOBB(cluster, bbox)) {
            ++rejected_geom;
            continue;
        }
        boxes.push_back(bbox);
    }

    spdlog::debug("[cluster_extractor] {} bboxes kept (rejected: {} pts, {} geom, min_pts={})",
                  boxes.size(), rejected_pts, rejected_geom, min_points);
    return boxes;
}

// ===========================================================================
// createOBB()  —  2D-PCA OBB (XY plane, Z world-vertical)
// ===========================================================================

bool DynamicClusterExtractor::createOBB(
    const std::vector<Eigen::Vector4d>& cluster,
    BoundingBox& out_bbox) const
{
    const int N = static_cast<int>(cluster.size());

    Eigen::Vector2d mean2d = Eigen::Vector2d::Zero();
    double z_min =  1e9, z_max = -1e9;
    for (const auto& p : cluster) {
        mean2d += p.head<2>();
        z_min = std::min(z_min, p.z());
        z_max = std::max(z_max, p.z());
    }
    mean2d /= static_cast<double>(N);

    Eigen::Matrix2d cov2d = Eigen::Matrix2d::Zero();
    for (const auto& p : cluster) {
        Eigen::Vector2d d = p.head<2>() - mean2d;
        cov2d += d * d.transpose();
    }
    cov2d /= static_cast<double>(N);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov2d);
    const Eigen::Matrix2d R2d = solver.eigenvectors();

    Eigen::Vector2d local_min( 1e9,  1e9);
    Eigen::Vector2d local_max(-1e9, -1e9);
    for (const auto& p : cluster) {
        Eigen::Vector2d local = R2d.transpose() * (p.head<2>() - mean2d);
        local_min = local_min.cwiseMin(local);
        local_max = local_max.cwiseMax(local);
    }

    const Eigen::Vector2d size2d         = local_max - local_min;
    const Eigen::Vector2d center2d_local = 0.5 * (local_min + local_max);
    const Eigen::Vector2d center2d       = R2d * center2d_local + mean2d;

    Eigen::Matrix3d R3d = Eigen::Matrix3d::Identity();
    R3d.block<2,2>(0,0) = R2d;

    const double z_size   = z_max - z_min;
    const double z_center = 0.5 * (z_min + z_max);

    const Eigen::Vector3d size  (size2d.x(), size2d.y(), z_size);
    const Eigen::Vector3d center(center2d.x(), center2d.y(), z_center);

    const double min_dim = size.minCoeff();
    const double max_dim = size.maxCoeff();
    const double volume  = size.x() * size.y() * size.z();

    if (params_.bbox_min_extent > 0.0 && min_dim < params_.bbox_min_extent) {
        spdlog::debug("[cluster_extractor] bbox rejected: min_extent {:.3f} < {:.3f}",
                      min_dim, params_.bbox_min_extent);
        return false;
    }
    if (params_.bbox_max_extent < 1e8 && max_dim > params_.bbox_max_extent) {
        spdlog::debug("[cluster_extractor] bbox rejected: max_extent {:.3f} > {:.3f}",
                      max_dim, params_.bbox_max_extent);
        return false;
    }
    if (params_.bbox_min_volume > 0.0 && volume < params_.bbox_min_volume) {
        spdlog::debug("[cluster_extractor] bbox rejected: volume {:.4f} < {:.4f}",
                      volume, params_.bbox_min_volume);
        return false;
    }
    if (params_.bbox_max_volume < 1e8 && volume > params_.bbox_max_volume) {
        spdlog::debug("[cluster_extractor] bbox rejected: volume {:.4f} > {:.4f}",
                      volume, params_.bbox_max_volume);
        return false;
    }

    out_bbox = BoundingBox(size, center, R3d);
    return true;
}

// ===========================================================================
// contains_center()
//
// Returns true if `query` center lies inside `container` expanded by margin.
// Works in the local frame of `container` to respect its orientation.
// ===========================================================================

static bool contains_center(
    const BoundingBox& container,
    const BoundingBox& query,
    double margin)
{
    // Transform query center into container's local frame
    const Eigen::Vector3d delta      = query.get_center() - container.get_center();
    const Eigen::Vector3d local      = container.get_rotation().transpose() * delta;
    const Eigen::Vector3d half_size  = 0.5 * container.get_size();
    const Eigen::Vector3d limit      = half_size + Eigen::Vector3d::Constant(margin);

    return (std::abs(local.x()) <= limit.x() &&
            std::abs(local.y()) <= limit.y() &&
            std::abs(local.z()) <= limit.z());
}

// ===========================================================================
// merge_with_history()
//
// Core idea: if a large historical bbox (already transformed to current frame)
// contains the centers of one or more current small bboxes, those small bboxes
// are removed and replaced by the historical one (marked static).
//
// This handles the frequent case where a single static object (wall segment,
// parked vehicle, pillar) is over-segmented into multiple clusters in the
// current frame due to partial occlusion or point density variation.
//
// Parameters that control this behaviour (all tunable in config):
//   containment_margin  — expand historical bbox by this amount before
//                         testing containment (handles odometry drift).
//   merge_volume_ratio  — historical bbox must be at least this times larger
//                         than the current one to trigger a merge.  Avoids
//                         replacing a large current bbox with an equally-large
//                         historical one that happens to share the same volume.
//
// Returns the merged bbox list. Historical bboxes that absorbed at least one
// current bbox are inserted once; absorbed current bboxes are dropped.
// Current bboxes not absorbed by any historical bbox are kept unchanged.
// ===========================================================================

std::vector<BoundingBox> DynamicClusterExtractor::merge_with_history(
    const std::vector<BoundingBox>& current_bboxes,
    const std::deque<std::vector<BoundingBox>>& history) const
{
    const int N = static_cast<int>(current_bboxes.size());

    // absorbed[i] = true means current_bboxes[i] has been swallowed
    std::vector<bool> absorbed(N, false);

    // Merged bboxes to add (historical bboxes that absorbed ≥1 current bbox)
    std::vector<BoundingBox> merged_in;

    const double margin       = params_.containment_margin;
    const double volume_ratio = params_.merge_volume_ratio;

    for (const auto& frame_bboxes : history) {
        for (const auto& hist_bbox : frame_bboxes) {

            const Eigen::Vector3d& hs = hist_bbox.get_size();
            const double hist_volume  = hs.x() * hs.y() * hs.z();

            std::vector<int> contained_ids;

            for (int i = 0; i < N; ++i) {
                if (absorbed[i]) continue;

                const Eigen::Vector3d& cs = current_bboxes[i].get_size();
                const double curr_volume  = cs.x() * cs.y() * cs.z();

                // Historical bbox must be meaningfully larger
                if (hist_volume < volume_ratio * curr_volume) continue;

                if (contains_center(hist_bbox, current_bboxes[i], margin))
                    contained_ids.push_back(i);
            }

            if (!contained_ids.empty()) {
                // Mark all contained current bboxes as absorbed
                for (int idx : contained_ids) absorbed[idx] = true;

                // Insert the historical bbox once, marked as static
                BoundingBox replacement = hist_bbox;
                replacement.set_dynamic(false);
                merged_in.push_back(replacement);

                spdlog::debug("[merge_with_history] hist bbox at ({:.2f},{:.2f},{:.2f}) "
                              "absorbed {} current bbox(es)",
                              hist_bbox.get_center().x(),
                              hist_bbox.get_center().y(),
                              hist_bbox.get_center().z(),
                              contained_ids.size());
            }
        }
    }

    // Build output: keep non-absorbed current bboxes + merged historical ones
    std::vector<BoundingBox> result;
    result.reserve(N);
    for (int i = 0; i < N; ++i)
        if (!absorbed[i]) result.push_back(current_bboxes[i]);

    result.insert(result.end(), merged_in.begin(), merged_in.end());

    spdlog::debug("[merge_with_history] {} in -> {} out ({} absorbed, {} merged in)",
                  N, result.size(),
                  N - static_cast<int>(std::count(absorbed.begin(), absorbed.end(), false)),
                  merged_in.size());

    return result;
}

// ===========================================================================
// classify_clusters()
//
// Pipeline:
//   1. Transform history to current frame.
//   2. merge_with_history(): replace over-segmented clusters with historical
//      large bboxes — handles static object fragmentation.
//   3. Standard static/dynamic classification via min_static_history_matches.
//   4. Update history.
// ===========================================================================

void DynamicClusterExtractor::classify_clusters(
    Eigen::Isometry3d T_delta_pose,
    std::vector<BoundingBox>& cluster_bboxes)
{
    // 1. Bring history into current sensor frame
    const Eigen::Isometry3d T_to_current = T_delta_pose.inverse();
    for (auto& frame_bboxes : cluster_history_)
        for (auto& bbox : frame_bboxes)
            bbox.transform(T_to_current);

    // 2. Temporal merging: absorb over-segmented clusters under large historical bboxes
    cluster_bboxes = merge_with_history(cluster_bboxes, cluster_history_);

    // 3. Standard static/dynamic classification on remaining bboxes
    const int available_history = static_cast<int>(cluster_history_.size());
    const int required_matches  = std::max(1,
        std::min(params_.min_static_history_matches, available_history));

    for (auto& cluster : cluster_bboxes) {
        // Already classified by merge_with_history — skip
        if (!cluster.is_dynamic_bbox()) continue;

        int static_matches = 0;

        for (const auto& frame_bboxes : cluster_history_) {
            for (const auto& hist_bbox : frame_bboxes) {
                const double dist = (cluster.get_center() - hist_bbox.get_center()).norm();
                const double iou  = cluster.iou(hist_bbox);

                if (iou  > params_.cluster_iou_threshold &&
                    dist < params_.cluster_distance_threshold) {
                    ++static_matches;
                    break;  // one match per frame
                }
            }
            if (static_matches >= required_matches) break;
        }

        const bool is_static = (static_matches >= required_matches);
        cluster.set_dynamic(!is_static);

        spdlog::debug("[cluster_extractor] cluster at ({:.2f},{:.2f},{:.2f}) -> {} "
                      "(static_matches={}/{})",
                      cluster.get_center().x(),
                      cluster.get_center().y(),
                      cluster.get_center().z(),
                      is_static ? "STATIC" : "DYNAMIC",
                      static_matches, required_matches);
    }

    // 4. Update history
    cluster_history_.push_front(cluster_bboxes);
    while (static_cast<int>(cluster_history_.size()) > params_.history_size)
        cluster_history_.pop_back();
}

} // namespace glim