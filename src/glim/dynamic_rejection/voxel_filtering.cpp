#include <glim/dynamic_rejection/voxel_filtering.hpp>

#include <algorithm>
#include <cmath>

#include <glim/preprocess/preprocessed_frame.hpp>
#include <glim/util/config.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/ann/impl/incremental_voxelmap_impl.hpp>

#include <spdlog/spdlog.h>

namespace glim {


WallFilterConfig::WallFilterConfig() {
    spdlog::debug("[wall_filter] WallFilterConfig::WallFilterConfig begin");
    Config config(GlobalConfig::get_config_path("config_wall_filter"));

    // voxel resolution must match odometry to reuse the same voxelmap
    voxel_resolution        = config.param<double>("wall_filter", "voxel_resolution",        0.5);
    ransac_max_iterations   = config.param<int>   ("wall_filter", "ransac_max_iterations",   500);
    ransac_inlier_threshold = config.param<double>("wall_filter", "ransac_inlier_threshold", 0.15);
    ransac_min_inliers      = config.param<int>   ("wall_filter", "ransac_min_inliers",      8);
    ransac_confidence       = config.param<double>("wall_filter", "ransac_confidence",       0.99);
    wall_vertical_angle_deg = config.param<double>("wall_filter", "wall_vertical_angle_deg", 20.0);
    max_planes              = config.param<int>   ("wall_filter", "max_planes",              8);
    floor_ceiling_angle_deg = config.param<double>("wall_filter", "floor_ceiling_angle_deg", 2.0);

    spdlog::debug("[wall_filter] WallFilterConfig: res={} ransac_iter={} thresh={} min_inliers={} "
                  "conf={} angle_deg={} max_planes={}",
                  voxel_resolution, ransac_max_iterations, ransac_inlier_threshold,
                  ransac_min_inliers, ransac_confidence, wall_vertical_angle_deg, max_planes);
}

WallFilterConfig::~WallFilterConfig() {
    spdlog::debug("[wall_filter] WallFilterConfig::~WallFilterConfig");
}

// ---------------------------------------------------------------------------
WallFilter::WallFilter(const WallFilterConfig& config, WallBBoxRegistry::Ptr bbox_registry)
    : config_(config), rng_(std::random_device{}()), bbox_registry_(bbox_registry) {}

// ---------------------------------------------------------------------------
WallFilterResult WallFilter::filter(const PreprocessedFrame& frame) {

    WallFilterResult result;

    // ------------------------------------------------------------------
    // Step 1 – Unica voxelizzazione del frame
    // Stesso pattern di DynamicObjectRejectionCPU::dynamic_object_rejection()
    // ------------------------------------------------------------------
    auto voxelmap = std::make_shared<gtsam_points::DynamicVoxelMapCPU>(
        config_.voxel_resolution);

    if (!frame.points.empty()) {
        auto pc = std::make_shared<gtsam_points::PointCloudCPU>(frame.points);
        if (!frame.intensities.empty()) 
            pc->add_intensities(frame.intensities);
        if (!frame.times.empty())       
            pc->add_times(frame.times);
        voxelmap->insert(*pc);
    }

    const int nvox = static_cast<int>(
        voxelmap->gtsam_points::IncrementalVoxelMap<
            gtsam_points::DynamicGaussianVoxel>::num_voxels());

    result.num_total_voxels = nvox;
    result.voxelmap         = voxelmap;

    spdlog::debug("[wall_filter] {} points → {} voxels", frame.points.size(), nvox);

    if (nvox < 3) {
        // Troppo pochi voxel per RANSAC: la voxelmap viene restituita intatta,
        // nessun voxel marcato come parete.
        return result;
    }

    // ------------------------------------------------------------------
    // Step 2 – Estrai centroidi dei voxel
    // ------------------------------------------------------------------
    std::vector<Eigen::Vector3d> centroids = extract_centroids(*voxelmap, nvox);

    // ------------------------------------------------------------------
    // Step 3 – RANSAC iterativo sui centroidi
    // Ad ogni iterazione il piano migliore viene estratto e i suoi inlier
    // rimossi dal set, permettendo di trovare più piani distinti.
    // ------------------------------------------------------------------
    std::vector<PlaneModel> all_planes;
    std::vector<Eigen::Vector3d> remaining = centroids;

    for (int p = 0; p < config_.max_planes; ++p) {
        if (static_cast<int>(remaining.size()) < config_.ransac_min_inliers) break;
        PlaneModel plane;
        if (!ransac_once(remaining, plane)) break;
        all_planes.push_back(plane);
    }

    spdlog::debug("[wall_filter] RANSAC: {} planes extracted", all_planes.size());

    // ------------------------------------------------------------------
    // Step 4 – Classifica i piani come parete o pavimento/soffitto
    // ------------------------------------------------------------------
    for (const auto& plane : all_planes) {
        if (is_wall_plane(plane) || is_floor_ceiling_plane(plane)) {
            result.wall_planes.push_back(plane);
            spdlog::debug("[wall_filter] wall plane: normal=({:.2f},{:.2f},{:.2f})",
                          plane.normal.x(), plane.normal.y(), plane.normal.z());
        }
    }

    spdlog::debug("[wall_filter] {} wall planes", result.wall_planes.size());

    if (result.wall_planes.empty()) {
        return result;
    }



    // ------------------------------------------------------------------
    // Step 5 – Marca i voxel-parete nella voxelmap
    // I voxel il cui centroide cade entro ransac_inlier_threshold da un
    // piano-parete vengono marcati con is_wall = true.
    // DynamicObjectRejectionCPU li skippera nel loop di dynamic scoring.
    // ------------------------------------------------------------------
    // result.num_wall_voxels = mark_wall_voxels(*voxelmap, nvox, result.wall_planes);

    spdlog::debug("[wall_filter] marked {}/{} voxels as walls",
                  result.num_wall_voxels, nvox);

    
    if (bbox_registry_ && !result.wall_planes.empty()) {
        const std::vector<Eigen::Vector3d> all_centroids =
            extract_centroids(*voxelmap, nvox);

        std::vector<BoundingBox> wall_bboxes;
        wall_bboxes.reserve(result.wall_planes.size());

        for (const auto& plane : result.wall_planes) {
            std::vector<Eigen::Vector3d> inliers;
            for (const auto& c : all_centroids) {
                if (plane.distance(c) < config_.ransac_inlier_threshold) {
                    inliers.push_back(c);
                }
            }
            spdlog::debug("[wall_filter] plane with {} inliers", inliers.size());
            if (static_cast<int>(inliers.size()) >= config_.ransac_min_inliers) {
                wall_bboxes.push_back(build_wall_bbox(inliers, plane));
            }
        }

        
        if (!wall_bboxes.empty()) {
            bbox_registry_->update(wall_bboxes);
            spdlog::debug("[wall_filter] bbox_registry updated with {} wall OBBs",
                          wall_bboxes.size());
        }
    }
    
    if (bbox_registry_ && !bbox_registry_->bboxes().empty()) {
        int registry_marked = 0;
        for (int i = 0; i < nvox; ++i) {
            auto& voxel = voxelmap->lookup_voxel(i);
            if (voxel.is_wall) continue;

            for (const auto& bbox : bbox_registry_->bboxes()) {
                if (bbox.contains(voxel.mean)) {
                    voxel.is_wall = true;
                    auto coord = voxelmap->voxel_coord(voxel.mean);
                    spdlog::debug("[wall_filter] voxel coordinates [{}, {}, {}]", coord.x(), coord.y(), coord.z());
                    ++registry_marked;
                    break;
                }
            }
        }
        spdlog::debug("[wall_filter] registry marked additional {}/{} voxels as walls",
                      registry_marked, nvox);
        result.num_wall_voxels += registry_marked;
    }

    
    return result;
}

// ---------------------------------------------------------------------------
std::vector<Eigen::Vector3d> WallFilter::extract_centroids(
    const gtsam_points::DynamicVoxelMapCPU& voxelmap, int nvox) const
{
    std::vector<Eigen::Vector3d> centroids;
    centroids.reserve(nvox);
    for (int i = 0; i < nvox; ++i) {
        // voxelmap.point(i) restituisce il mean del voxel (Eigen::Vector4d omogeneo)
        auto& voxel = voxelmap.lookup_voxel(i);
        const Eigen::Vector4d& pt = voxel.mean;
        centroids.emplace_back(pt.x(), pt.y(), pt.z());
    }
    return centroids;
}

// ---------------------------------------------------------------------------
PlaneModel WallFilter::fit_plane(const Eigen::Vector3d& p0,
                                  const Eigen::Vector3d& p1,
                                  const Eigen::Vector3d& p2) const
{
    Eigen::Vector3d n    = (p1 - p0).cross(p2 - p0);
    const double    norm = n.norm();

    PlaneModel plane;
    if (norm < 1e-9) {
        // Caso degenere: normale verticale → non sarà classificato parete
        plane.normal = Eigen::Vector3d::UnitZ();
        plane.d      = -plane.normal.dot(p0);
        return plane;
    }
    plane.normal = n / norm;
    plane.d      = -plane.normal.dot(p0);
    return plane;
}

// ---------------------------------------------------------------------------
std::vector<int> WallFilter::find_inliers(
    const std::vector<Eigen::Vector3d>& pts,
    const PlaneModel& plane,
    double threshold) const
{
    std::vector<int> inliers;
    inliers.reserve(pts.size() / 4);
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        if (std::abs(plane.normal.dot(pts[i]) + plane.d) < threshold)
            inliers.push_back(i);
    }
    return inliers;
}

// ---------------------------------------------------------------------------
PlaneModel WallFilter::refit_plane(const std::vector<Eigen::Vector3d>& pts,
                                    const std::vector<int>& inlier_idx) const
{
    const int m = static_cast<int>(inlier_idx.size());

    // Centroide
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (int idx : inlier_idx) mean += pts[idx];
    mean /= static_cast<double>(m);

    // Matrice di covarianza
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int idx : inlier_idx) {
        Eigen::Vector3d d = pts[idx] - mean;
        cov += d * d.transpose();
    }

    // Normale = autovettore con autovalore MINIMO (minima varianza perpendicolare)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d normal = solver.eigenvectors().col(0).normalized();

    PlaneModel plane;
    plane.normal = normal;
    plane.d      = -normal.dot(mean);
    return plane;
}

// ---------------------------------------------------------------------------
bool WallFilter::ransac_once(std::vector<Eigen::Vector3d>& pts, PlaneModel& best_plane) {
    const int n = static_cast<int>(pts.size());
    if (n < 3) return false;

    std::uniform_int_distribution<int> dist(0, n - 1);

    int              best_count = 0;
    std::vector<int> best_inliers;

    for (int iter = 0; iter < config_.ransac_max_iterations; ++iter) {
        // Campiona 3 punti distinti
        int i0, i1, i2;
        i0 = dist(rng_);
        do { i1 = dist(rng_); } while (i1 == i0);
        do { i2 = dist(rng_); } while (i2 == i0 || i2 == i1);

        PlaneModel candidate = fit_plane(pts[i0], pts[i1], pts[i2]);
        if (candidate.normal.norm() < 0.5) continue;  // degenerato

        std::vector<int> inliers =
            find_inliers(pts, candidate, config_.ransac_inlier_threshold);

        if (static_cast<int>(inliers.size()) > best_count) {
            best_count   = static_cast<int>(inliers.size());
            best_inliers = std::move(inliers);
            best_plane   = candidate;

            // Terminazione adattiva: N = log(1-p) / log(1-w^3)
            const double w      = static_cast<double>(best_count) / n;
            const double w3     = w * w * w;
            if (w3 > 1.0 - 1e-9) break;
            const double n_iter = std::log(1.0 - config_.ransac_confidence)
                                / std::log(1.0 - w3 + 1e-10);
            if (iter + 1 >= static_cast<int>(std::ceil(n_iter))) break;
        }
    }

    if (best_count < config_.ransac_min_inliers) return false;

    // Refit finale con tutti gli inlier
    best_plane = refit_plane(pts, best_inliers);

    // Rimuovi inlier da pts per la prossima iterazione RANSAC
    std::vector<bool> is_inlier(n, false);
    for (int idx : best_inliers) is_inlier[idx] = true;

    std::vector<Eigen::Vector3d> remaining;
    remaining.reserve(n - best_count);
    for (int i = 0; i < n; ++i)
        if (!is_inlier[i]) remaining.push_back(pts[i]);

    pts = std::move(remaining);
    return true;
}

// ---------------------------------------------------------------------------
bool WallFilter::is_wall_plane(const PlaneModel& plane) const {
    // Parete → normale quasi orizzontale → |normal.z| piccolo
    // Pavimento/soffitto → normale quasi verticale → |normal.z| ≈ 1
    //
    // |normal.z| = sin(angolo tra normale e piano XY)
    // Condizione parete: angolo < wall_vertical_angle_deg
    //                  → |normal.z| < sin(wall_vertical_angle_deg)
    const double sin_thr = std::sin(config_.wall_vertical_angle_deg * M_PI / 180.0);
    return std::abs(plane.normal.z()) < sin_thr;
}

bool WallFilter::is_floor_ceiling_plane(const PlaneModel& plane) const {
    // Pavimento/soffitto → normale quasi verticale → |normal.z| vicino a 1
    // Condizione: angolo tra normale e asse Z < floor_ceiling_angle_deg
    //           → |normal.z| > cos(floor_ceiling_angle_deg)
    const double cos_thr = std::cos(config_.floor_ceiling_angle_deg * M_PI / 180.0);
    return std::abs(plane.normal.z()) > cos_thr;
}

// ---------------------------------------------------------------------------
int WallFilter::mark_wall_voxels(
    gtsam_points::DynamicVoxelMapCPU& voxelmap,
    int nvox,
    const std::vector<PlaneModel>& wall_planes) const
{
    int count = 0;
    for (int i = 0; i < nvox; ++i) {
        auto& voxel = voxelmap.lookup_voxel(i);
        const Eigen::Vector3d c(voxel.mean.x(), voxel.mean.y(), voxel.mean.z());

        for (const auto& plane : wall_planes) {
            if (plane.distance(c) < config_.ransac_inlier_threshold) {
                voxel.is_wall = true;
                ++count;
                break;  // basta un piano
            }
        }
    }
    return count;
}


BoundingBox WallFilter::build_wall_bbox(
    const std::vector<Eigen::Vector3d>& inlier_pts,
    const PlaneModel& plane) const
{
    if (inlier_pts.empty()) {
        return BoundingBox(Eigen::Vector3d::Zero(),
                           Eigen::Vector3d::Zero(),
                           Eigen::Matrix3d::Identity());
    }

    // Usa direttamente gli assi del piano senza normalizzare verso UnitZ
    const Eigen::Vector3d n = plane.normal.normalized();

    // Asse arbitrario non parallelo a n per costruire il frame
    Eigen::Vector3d ref = Eigen::Vector3d::UnitX();
    if (std::abs(n.dot(ref)) > 0.9) {
        ref = Eigen::Vector3d::UnitY();
    }

    // Gram-Schmidt: right ortogonale a n, up ortogonale a entrambi
    const Eigen::Vector3d right = (ref - n.dot(ref) * n).normalized();
    const Eigen::Vector3d up    = n.cross(right).normalized();

    Eigen::Matrix3d R;
    R.col(0) = n;
    R.col(1) = right;
    R.col(2) = up;

    // ── Proietta tutti i centroidi nel frame locale ───────────────────────────
    // Usiamo il primo punto come origine temporanea per stabilità numerica
    const Eigen::Vector3d origin = inlier_pts[0];

    Eigen::Vector3d local_min( 1e9,  1e9,  1e9);
    Eigen::Vector3d local_max(-1e9, -1e9, -1e9);

    for (const auto& p : inlier_pts) {
        const Eigen::Vector3d local = R.transpose() * (p - origin);
        local_min = local_min.cwiseMin(local);
        local_max = local_max.cwiseMax(local);
    }

    // ── Dimensioni ────────────────────────────────────────────────────────────
    Eigen::Vector3d size = local_max - local_min;

    // Lo spessore lungo la normale (asse 0) è artificialmente piccolo
    // perché i centroidi sono quasi complanari: imponiamo un minimo
    size.x() = std::max(size.x(), config_.ransac_inlier_threshold * 2.0);

    // ── Centro nel frame locale → mondo ──────────────────────────────────────
    const Eigen::Vector3d local_center = 0.5 * (local_max + local_min);
    const Eigen::Vector3d world_center = R * local_center + origin;

    return BoundingBox(size, world_center, R);
}



}  // namespace glim