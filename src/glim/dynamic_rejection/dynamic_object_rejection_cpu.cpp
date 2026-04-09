#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>

#include <algorithm>
#include <cmath>

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <spdlog/spdlog.h>

#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>
#include <gtsam_points/util/parallelism.hpp>
#include <gtsam_points/config.hpp>

#include <glim/util/config.hpp>

#ifdef GTSAM_POINTS_USE_TBB
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#endif

#include <glim/dynamic_rejection/bounding_box.hpp>

namespace glim {

// ===========================================================================
// Helper: num_voxels() from a DynamicVoxelMapCPU
// ===========================================================================

static int nvox_of(const gtsam_points::DynamicVoxelMapCPU& vm) {
    return static_cast<int>(
        vm.gtsam_points::IncrementalVoxelMap<
            gtsam_points::DynamicGaussianVoxel>::num_voxels());
}

// ===========================================================================
// DynamicObjectRejectionParamsCPU
// ===========================================================================

DynamicObjectRejectionParamsCPU::DynamicObjectRejectionParamsCPU() {
    Config config(GlobalConfig::get_config_path("config_dynamic_object_rejection"));

    dynamic_score_threshold  = config.param<double>("dynamic_object_rejection", "dynamic_score_threshold",  2.5);
    num_threads              = config.param<int>   ("dynamic_object_rejection", "num_threads",              4);
    w_shift                  = config.param<double>("dynamic_object_rejection", "w_shift",                  1.0);
    w_mahalanobis            = config.param<double>("dynamic_object_rejection", "w_mahalanobis",            0.8);
    w_covariance_difference  = config.param<double>("dynamic_object_rejection", "w_covariance_difference",  0.5);
    w_shape                  = config.param<double>("dynamic_object_rejection", "w_shape",                  0.5);
    w_occupancy              = config.param<double>("dynamic_object_rejection", "w_occupancy",              0.3);
    w_cluster                = config.param<double>("dynamic_object_rejection", "w_cluster",                0.07);
    w_history                = config.param<double>("dynamic_object_rejection", "w_history",                0.09);
    w_neighbor               = config.param<double>("dynamic_object_rejection", "w_neighbor",               0.6);
    history_factor           = config.param<double>("dynamic_object_rejection", "history_factor",           0.5);
    frame_num_memory         = config.param<int>   ("dynamic_object_rejection", "frame_num_memory",         10);
    points_limit             = config.param<double>("dynamic_object_rejection", "points_limit",             0.25);
    cluster_propagation_threshold = config.param<double>("dynamic_object_rejection", "cluster_propagation_threshold", 0.5);


    spdlog::debug("[dynamic_rejection] params loaded");
}

DynamicObjectRejectionParamsCPU::~DynamicObjectRejectionParamsCPU() = default;


// ===========================================================================
// Construction
// ===========================================================================

DynamicObjectRejectionCPU::DynamicObjectRejectionCPU(
    const DynamicObjectRejectionParamsCPU&  params,
    const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter)
    : params_(params), pose_kalman_filter_(pose_kalman_filter), last_pose_(Eigen::Isometry3d::Identity())
{
    covariance_estimation_ = std::make_unique<CloudCovarianceEstimation>(params_.num_threads);

    if (!pose_kalman_filter_) {
        pose_kalman_filter_ = std::make_shared<PoseKalmanFilter>();
    }

    spdlog::debug("[dynamic_rejection] DynamicObjectRejectionCPU constructed");
}


// ===========================================================================
// reject()  —  primary API
// ===========================================================================

DynamicRejectionResult DynamicObjectRejectionCPU::reject(
    const WallFilterResult&       wf_result,
    const PreprocessedFrame::Ptr& source_frame,
    const std::vector<BoundingBox>& cluster_bboxes)
{
    spdlog::debug("[dynamic_rejection] reject begin: voxelmap={} wall_voxels={}/{}",
        static_cast<bool>(wf_result.voxelmap),
        wf_result.num_wall_voxels,
        wf_result.num_total_voxels);

    DynamicRejectionResult result;
    result.voxelmap = wf_result.voxelmap;

    // -----------------------------------------------------------------------
    // First frame: store as reference, return source unchanged
    // -----------------------------------------------------------------------
    if (voxelmap_history_.empty()) {
        spdlog::debug("[dynamic_rejection] first frame — storing reference voxelmap");
        voxelmap_history_.push_back(wf_result.voxelmap);
        pose_history_.push_back(pose_kalman_filter_->getPose());
        last_dynamic_frame_ = nullptr;
        result.static_frame  = source_frame;
        result.dynamic_frame = nullptr;
        return result;
    }

    Eigen::Isometry3d cur_pose = pose_kalman_filter_->getPose();
    Eigen::Isometry3d T_delta_pose =  cur_pose * last_pose_.inverse();
    last_pose_ = cur_pose;

    spdlog::debug("[dynamic_rejection] delta pose since last frame: translation=({:.4f},{:.4f},{:.4f})",
        T_delta_pose.translation().x(), T_delta_pose.translation().y(), T_delta_pose.translation().z());

    // -----------------------------------------------------------------------
    // Score voxels against previous frame, then propagate to neighbours
    // -----------------------------------------------------------------------
    
    auto t0 = std::chrono::steady_clock::now();
    score_voxels(*wf_result.voxelmap, *voxelmap_history_.back(), cluster_bboxes, T_delta_pose);
    auto t1 = std::chrono::steady_clock::now();
    spdlog::debug("[dynamic_rejection] score_voxels took {:.1f} ms",
    std::chrono::duration<double, std::milli>(t1 - t0).count());

    const int nvox = nvox_of(*wf_result.voxelmap);
    propagate_to_neighbors(*wf_result.voxelmap, nvox);
    propagate_to_clusters(*wf_result.voxelmap, cluster_bboxes);

    // -----------------------------------------------------------------------
    // Split voxel points into static / dynamic buckets
    // -----------------------------------------------------------------------
    std::vector<Eigen::Vector4d> static_pts,  dynamic_pts;
    std::vector<double>          static_int,  dynamic_int;
    std::vector<double>          static_tim,  dynamic_tim;

    collect_points(*wf_result.voxelmap,
                   static_pts,  static_int,  static_tim,
                   dynamic_pts, dynamic_int, dynamic_tim);

    // -----------------------------------------------------------------------
    // Update history ring
    // -----------------------------------------------------------------------
    voxelmap_history_.push_back(wf_result.voxelmap);
    if (static_cast<int>(voxelmap_history_.size()) > params_.frame_num_memory) {
        voxelmap_history_.erase(voxelmap_history_.begin());
    }
    pose_history_.push_back(T_delta_pose);
    if (static_cast<int>(pose_history_.size()) > params_.frame_num_memory) {
        pose_history_.erase(pose_history_.begin());
    }

    // -----------------------------------------------------------------------
    // Build output frames
    // -----------------------------------------------------------------------
    if (static_pts.empty()) {
        spdlog::warn("[dynamic_rejection] no static points — returning source frame unchanged");
        result.static_frame = source_frame;
    } else {
        result.static_frame = build_frame(
            std::move(static_pts), std::move(static_int), std::move(static_tim),
            *source_frame);
    }

    if (!dynamic_pts.empty()) {
        auto dyn           = std::make_shared<PreprocessedFrame>();
        dyn->stamp         = source_frame->stamp;
        dyn->scan_end_time = source_frame->scan_end_time;
        dyn->points        = std::move(dynamic_pts);
        dyn->intensities   = std::move(dynamic_int);
        dyn->times         = std::move(dynamic_tim);
        dyn->k_neighbors   = 0;
        last_dynamic_frame_  = dyn;
        result.dynamic_frame = dyn;
    } else {
        last_dynamic_frame_  = nullptr;
        result.dynamic_frame = nullptr;
    }

    spdlog::debug("[dynamic_rejection] reject done: static={} dynamic={}",
        result.static_frame  ? result.static_frame->points.size()  : 0,
        result.dynamic_frame ? result.dynamic_frame->points.size() : 0);

    return result;
}


// ===========================================================================
// score_voxels()
// ===========================================================================

void DynamicObjectRejectionCPU::score_voxels(
    gtsam_points::DynamicVoxelMapCPU&       current,
    const gtsam_points::DynamicVoxelMapCPU& previous,
    const std::vector<BoundingBox>&         cluster_bboxes,
    const Eigen::Isometry3d&                T_delta_pose)
{
    const int nvox = nvox_of(current);
    dynamic_voxels_neighbor_indices_.clear();
    dynamic_voxels.clear();
    for (int j = 0; j < nvox; ++j) {
        auto& cur = current.lookup_voxel(j);

        // Wall voxels are already marked by WallFilter — keep as static
        if (cur.is_wall) {
            cur.is_dynamic    = false;
            cur.dynamic_score = 0.0;
            continue;
        }

        // Too few points to compare reliably — keep as static
        if (cur.num_points/(current.voxel_resolution()*100) < params_.points_limit) {
            cur.is_dynamic    = false;
            cur.dynamic_score = 0.0;
            continue;
        }
        
        cur.is_dynamic    = false;
        cur.dynamic_score = 0.0;
        bool in_dynamic_cluster_bbox = false;
        bool possibly_dynamic = false;
        for (const auto& cluster : cluster_bboxes) {
            if (cluster.contains(cur.mean)) {
                in_dynamic_cluster_bbox = true;    
                if (!cluster.is_dynamic_bbox()) {
                    spdlog::debug("[dynamic_rejection] voxel {:3d}: suppressed by cluster bbox", j);
                    cur.dynamic_score += -params_.w_cluster/2.0;
                }
                else {
                    spdlog::debug("[dynamic_rejection] voxel {:3d}: boosted by dynamic cluster bbox", j);
                    cur.dynamic_score += params_.w_cluster;
                    possibly_dynamic = true;
                }
                break;
            }
        }

        if (!in_dynamic_cluster_bbox) {
            cur.dynamic_score += -params_.w_cluster/2.0;
        }

        Eigen::Vector3d p_trans = T_delta_pose * cur.mean.head<3>();
        Eigen::Vector4d p_trans4(p_trans.x(), p_trans.y(), p_trans.z(), 1.0);
        const auto base = previous.voxel_coord(p_trans4);
        const auto coord    = current.voxel_coord(p_trans4);
        const int  prev_idx = previous.lookup_voxel_index(coord);


        if (prev_idx < 0) {
            // Voxel newly appeared — flag as dynamic candidate
            if (!possibly_dynamic) {
                spdlog::debug("[dynamic_rejection] voxel {:3d}: new voxel but not in dynamic cluster bbox, score={:.3f}", j, cur.dynamic_score);
                cur.is_dynamic    = false;
                cur.dynamic_score = 0.0;
            }
            else{
                spdlog::debug("[dynamic_rejection] voxel {:3d}: new voxel, marking as dynamic candidate, score={:.3f}", j, cur.dynamic_score);
                cur.is_dynamic    = true;
                cur.dynamic_score = params_.dynamic_score_threshold + 1.0;
                const auto nbrs = get_neighbor_voxels(current, cur.mean);
                dynamic_voxels.insert(dynamic_voxels.end(), nbrs.size(), j);
                dynamic_voxels_neighbor_indices_.insert(dynamic_voxels_neighbor_indices_.end(), nbrs.begin(), nbrs.end());    
            }
            continue;
        }

        const auto& prv = previous.lookup_voxel(prev_idx);

        cur.is_dynamic    = false;
        cur.dynamic_score = 0.0;

        // ---- 1. Centroid shift (normalized by voxel size) ----
        const Eigen::Vector3d delta = (p_trans - prv.mean.head<3>());
        const double shift = delta.norm() / current.voxel_resolution();

        // ---- 2. Mahalanobis distance ----
        double mahal = 0.0;
        
        {
            Eigen::Matrix3d cov = prv.cov.topLeftCorner<3,3>();
            cov = 0.5 * (cov + cov.transpose());
            cov.diagonal().array() += 1e-6;

            Eigen::LDLT<Eigen::Matrix3d> ldlt(cov);
            if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
                const Eigen::Vector3d sol = ldlt.solve(delta);
                if (sol.allFinite()) {
                    const double q = delta.dot(sol);
                    if (std::isfinite(q) && q >= 0.0) mahal = std::sqrt(q);
                }
            }
        }

        // ---- 3. Covariance Frobenius difference ----
        const double cov_diff =
            (cur.cov - prv.cov).norm() /
            (cur.cov.norm() + prv.cov.norm() + 1e-6);

        // ---- 4. Eigenvalue shape change ----
        double shape_change = 0.0;
        {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> ep(prv.cov.topLeftCorner<3,3>());
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> ec(cur.cov.topLeftCorner<3,3>());
            if (ep.info() == Eigen::Success && ec.info() == Eigen::Success) {
                shape_change = (ep.eigenvalues() - ec.eigenvalues()).norm() /
                               (ep.eigenvalues().norm() + 1e-6);
            }
        }

        // ---- 5. Occupancy ratio ----
        const double occ_ratio =
            std::abs(static_cast<double>(cur.num_points) -
                     static_cast<double>(prv.num_points)) /
            (cur.num_points + prv.num_points + 1e-6);

        // ---- Aggregate ----
        cur.dynamic_score =
            params_.w_shift                 * shift        +
            params_.w_mahalanobis           * mahal        +
            params_.w_covariance_difference * cov_diff     +
            params_.w_shape                 * shape_change +
            params_.w_occupancy             * occ_ratio;

        spdlog::debug("[dynamic_rejection] voxel {:3d}: score={:.3f} "
                      "(shift={:.3f} mahal={:.3f} cov={:.3f} shape={:.3f} occ={:.3f})",
                      j, cur.dynamic_score, shift, mahal, cov_diff, shape_change, occ_ratio);

        // ---- History suppression ----
        
        int static_count = 0;
        Eigen::Vector4d cur_mean = p_trans4;
        for (int k = 0; k < params_.frame_num_memory &&
                            k < static_cast<int>(voxelmap_history_.size()) &&
                            k < static_cast<int>(pose_history_.size()); ++k)
        {
            const int idx_hist = static_cast<int>(voxelmap_history_.size()) - 1 - k;
            const auto& past      = voxelmap_history_[idx_hist];
            const auto& pose_prev = pose_history_[idx_hist];

        
            int voxel_idx = past->lookup_voxel_index(past->voxel_coord(cur_mean));
            Eigen::Vector3d tmp = pose_prev * cur_mean.head<3>();
            cur_mean = Eigen::Vector4d(tmp.x(), tmp.y(), tmp.z(), 1.0);
            if (voxel_idx < 0) continue;

            
            ++static_count;
        }
        int seen_total = std::min(params_.frame_num_memory, static_cast<int>(voxelmap_history_.size()))-1;

        const double static_ratio =
            (seen_total > 0) ? static_cast<double>(static_count) / seen_total : 0.0;


        if (static_ratio > params_.history_factor) {
            spdlog::debug("[dynamic_rejection] voxel {:3d}: suppressed by history ({:.0f}% static)", j, static_ratio * 100.0);
            cur.dynamic_score -= params_.w_history;
            continue;
        }

        


        // ---- Threshold ----
        if (cur.dynamic_score > params_.dynamic_score_threshold) {
            cur.is_dynamic = true;
            const auto nbrs = get_neighbor_voxels(current, cur.mean);
            dynamic_voxels.insert(dynamic_voxels.end(), nbrs.size(), j);
            dynamic_voxels_neighbor_indices_.insert(dynamic_voxels_neighbor_indices_.end(), nbrs.begin(), nbrs.end());
            spdlog::debug("[dynamic_rejection] voxel {:3d}: DYNAMIC", j);
        } else {
            spdlog::debug("[dynamic_rejection] voxel {:3d}: static", j);
        }
    }
}

// ===========================================================================
// propagate_to_neighbors()
// ===========================================================================

void DynamicObjectRejectionCPU::propagate_to_neighbors(
    gtsam_points::DynamicVoxelMapCPU& voxelmap, int nvox)
{
    for (int idx : dynamic_voxels_neighbor_indices_) {
        if (idx < 0 || idx >= nvox) continue;

        auto& v = voxelmap.lookup_voxel(idx);
        if (v.is_dynamic || v.is_wall) continue;

        v.dynamic_score += params_.w_neighbor;
        if (v.dynamic_score > params_.dynamic_score_threshold) {
            v.is_dynamic = true;
        }
    }
    dynamic_voxels_neighbor_indices_.clear();
    dynamic_voxels.clear();
}



// ===========================================================================
// propagate_to_clusters()
// ===========================================================================
// If a percentage of voxels in a cluster are dynamic, mark the whole cluster as dynamic.
void DynamicObjectRejectionCPU::propagate_to_clusters(
    gtsam_points::DynamicVoxelMapCPU& voxelmap,
    const std::vector<BoundingBox>& cluster_bboxes)
{
    if (cluster_bboxes.empty()) return;

    const int nvox = nvox_of(voxelmap);
    const int n_clusters = static_cast<int>(cluster_bboxes.size());

    // Un solo passaggio: assegna ogni voxel al suo cluster (-1 = nessuno)
    std::vector<int>  voxel_cluster(nvox, -1);
    std::vector<int>  dynamic_count(n_clusters, 0);
    std::vector<int>  total_count(n_clusters, 0);

    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);
        for (int c = 0; c < n_clusters; ++c) {
            if (cluster_bboxes[c].contains(v.mean)) {
                voxel_cluster[j] = c;
                ++total_count[c];
                if (v.is_dynamic) ++dynamic_count[c];
                break; // un voxel può appartenere a un solo cluster
            }
        }
    }

    // Determina quali cluster vanno marcati dinamici
    std::vector<bool> cluster_is_dynamic(n_clusters, false);
    for (int c = 0; c < n_clusters; ++c) {
        if (total_count[c] == 0) continue;
        const double ratio = static_cast<double>(dynamic_count[c]) / total_count[c];
        cluster_is_dynamic[c] = (ratio > params_.cluster_propagation_threshold);
        spdlog::debug("[dynamic_rejection] cluster {}: {}/{} dynamic ({:.1f}%) -> {}",
                      c, dynamic_count[c], total_count[c], ratio * 100.0,
                      cluster_is_dynamic[c] ? "DYNAMIC" : "static");
    }

    // Secondo passaggio (solo se almeno un cluster è dinamico)
    bool any_dynamic = std::any_of(cluster_is_dynamic.begin(), cluster_is_dynamic.end(),
                                   [](bool b){ return b; });
    if (!any_dynamic) return;

    for (int j = 0; j < nvox; ++j) {
        const int c = voxel_cluster[j];
        if (c < 0 || !cluster_is_dynamic[c] ) {
            spdlog::debug("[dynamic_rejection] voxel {:3d}: cluster {} is not dynamic or not valid", j, c);
            continue;
        }
        auto& v = voxelmap.lookup_voxel(j);
        if (!v.is_wall)
                v.is_dynamic = true;
    }
}
// ===========================================================================
// collect_points()
// ===========================================================================

void DynamicObjectRejectionCPU::collect_points(
    const gtsam_points::DynamicVoxelMapCPU& voxelmap,
    std::vector<Eigen::Vector4d>& static_pts,
    std::vector<double>&          static_int,
    std::vector<double>&          static_tim,
    std::vector<Eigen::Vector4d>& dynamic_pts,
    std::vector<double>&          dynamic_int,
    std::vector<double>&          dynamic_tim) const
{
    const int nvox = nvox_of(voxelmap);
    for (int j = 0; j < nvox; ++j) {
        const auto& v = voxelmap.lookup_voxel(j);

        auto& pts  = v.is_dynamic ? dynamic_pts : static_pts;
        auto& ints = v.is_dynamic ? dynamic_int : static_int;
        auto& tims = v.is_dynamic ? dynamic_tim : static_tim;

        pts.insert(pts.end(),   v.voxel_points.begin(),      v.voxel_points.end());
        ints.insert(ints.end(), v.voxel_intensities.begin(), v.voxel_intensities.end());
        tims.insert(tims.end(), v.voxel_times.begin(),       v.voxel_times.end());
    }
}


// ===========================================================================
// build_frame()
// ===========================================================================

PreprocessedFrame::Ptr DynamicObjectRejectionCPU::build_frame(
    std::vector<Eigen::Vector4d> points,
    std::vector<double>          intensities,
    std::vector<double>          times,
    const PreprocessedFrame&     source) const
{
    auto out           = std::make_shared<PreprocessedFrame>();
    out->stamp         = source.stamp;
    out->scan_end_time = source.scan_end_time;
    out->k_neighbors   = source.k_neighbors;
    out->points        = std::move(points);
    out->intensities   = std::move(intensities);
    out->times         = std::move(times);

    if (source.k_neighbors > 0 && !out->points.empty()) {
        out->neighbors = find_neighbors(
            out->points.data(),
            static_cast<int>(out->points.size()),
            source.k_neighbors);
    }

    return out;
}


// ===========================================================================
// find_neighbors()
// ===========================================================================

std::vector<int> DynamicObjectRejectionCPU::find_neighbors(
    const Eigen::Vector4d* points, int num_points, int k) const
{
    if (!points || num_points <= 0 || k <= 0) return {};

    gtsam_points::KdTree tree(points, num_points);
    std::vector<int>     neighbors(num_points * k);

    const auto per_point = [&](int i) {
        std::vector<size_t> k_idx(k, static_cast<size_t>(i));
        std::vector<double> k_sq(k);
        const size_t found =
            tree.knn_search(points[i].data(), k, k_idx.data(), k_sq.data());
        std::copy(k_idx.begin(), k_idx.begin() + found,
                  neighbors.begin() + i * k);
    };

    if (gtsam_points::is_omp_default()) {
#pragma omp parallel for num_threads(params_.num_threads) schedule(guided, 8)
        for (int i = 0; i < num_points; ++i) per_point(i);
    } else {
#ifdef GTSAM_POINTS_USE_TBB
        tbb::parallel_for(
            tbb::blocked_range<int>(0, num_points, 8),
            [&](const tbb::blocked_range<int>& r) {
                for (int i = r.begin(); i < r.end(); ++i) per_point(i);
            });
#else
        for (int i = 0; i < num_points; ++i) per_point(i);
#endif
    }

    return neighbors;
}


// ===========================================================================
// get_neighbor_voxels()
// ===========================================================================

std::vector<int> DynamicObjectRejectionCPU::get_neighbor_voxels(
    const gtsam_points::DynamicVoxelMapCPU& voxelmap,
    const Eigen::Vector4d& mean) const
{
    std::vector<int> neighbors;
    const auto base = voxelmap.voxel_coord(mean);

    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        Eigen::Vector3i c = base;
        c.x() += dx; c.y() += dy; c.z() += dz;
        const int idx = voxelmap.lookup_voxel_index(c);
        if (idx >= 0) neighbors.push_back(idx);
    }

    return neighbors;
}

}  // namespace glim