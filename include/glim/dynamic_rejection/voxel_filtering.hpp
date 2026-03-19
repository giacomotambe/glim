#pragma once

#include <memory>
#include <vector>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <glim/dynamic_rejection/dynamic_voxelmap_cpu.hpp>  // adjust path

namespace glim {

struct PreprocessedFrame;

// ---------------------------------------------------------------------------
// Piano 3D: normal.dot(p) + d = 0,  ||normal|| = 1
// ---------------------------------------------------------------------------
struct PlaneModel {
    Eigen::Vector3d normal;
    double d;

    double distance(const Eigen::Vector3d& pt) const {
        return std::abs(normal.dot(pt) + d);
    }
};

// ---------------------------------------------------------------------------
// Risultato di WallFilter::filter()
// ---------------------------------------------------------------------------
struct WallFilterResult {
    /// Voxelmap contenente SOLO i voxel non-parete.
    /// Passato direttamente a DynamicObjectRejectionCPU (no re-voxelizzazione).
    gtsam_points::DynamicVoxelMapCPU::Ptr voxelmap;

    /// Piani-parete trovati da RANSAC (utile per debug/visualizzazione)
    std::vector<PlaneModel> wall_planes;

    /// Numero di voxel classificati come parete (rimossi dalla voxelmap)
    int num_wall_voxels = 0;

    /// Numero di voxel totali prima del filtraggio
    int num_total_voxels = 0;
};

// ---------------------------------------------------------------------------
// Configurazione
// ---------------------------------------------------------------------------
struct WallFilterConfig {
    // Voxelizzazione — deve coincidere con voxel_resolution dell'odometry
    double voxel_resolution = 0.5;

    // RANSAC
    int    ransac_max_iterations   = 500;
    double ransac_inlier_threshold = 0.15;  // [m] — usata anche per marcare i voxel
    int    ransac_min_inliers      = 8;
    double ransac_confidence       = 0.99;

    // Classificazione parete:
    // angolo massimo (gradi) tra la normale del piano e il piano orizzontale XY.
    // Pareti quasi verticali hanno normale quasi orizzontale → |nz| piccolo.
    // Es. 20° → accetta pareti inclinate al massimo di 20° rispetto alla verticale.
    double wall_vertical_angle_deg = 20.0;

    // Estrazione multi-piano
    int max_planes = 8;
};

// ---------------------------------------------------------------------------
// WallFilter
//
// Pipeline:
//   1. Voxelizza PreprocessedFrame → DynamicVoxelMapCPU (unica voxelizzazione)
//   2. Estrai centroidi voxel
//   3. RANSAC iterativo sui centroidi → piani dominanti
//   4. Classifica ogni piano: parete se normale quasi orizzontale
//   5. Marca i voxel-parete nella voxelmap (is_dynamic = true come sentinel,
//      oppure con flag is_wall se aggiunto a DynamicGaussianVoxel)
//   6. Restituisce WallFilterResult con la voxelmap già pronta per
//      DynamicObjectRejectionCPU — nessuna doppia voxelizzazione.
//
// DynamicObjectRejectionCPU deve essere adattato per accettare una
// voxelmap pre-costruita invece di un PreprocessedFrame raw.
// ---------------------------------------------------------------------------
class WallFilter {
public:
    using Ptr      = std::shared_ptr<WallFilter>;
    using ConstPtr = std::shared_ptr<const WallFilter>;

    explicit WallFilter(const WallFilterConfig& config = WallFilterConfig{});
    ~WallFilter() = default;

    /**
     * @brief  Voxelizza il frame, identifica i voxel-parete via RANSAC,
     *         e restituisce la voxelmap con i voxel-parete marcati.
     *
     * I voxel-parete sono marcati con is_wall = true nel DynamicGaussianVoxel.
     * DynamicObjectRejectionCPU deve saltarli nel loop di dynamic scoring
     * (vedi dynamic_object_recognition).
     *
     * @param frame  Frame preprocessato in ingresso.
     * @return       WallFilterResult con voxelmap + metadati.
     */
    WallFilterResult filter(const PreprocessedFrame& frame);

private:
    std::vector<Eigen::Vector3d> extract_centroids(
        const gtsam_points::DynamicVoxelMapCPU& voxelmap, int nvox) const;

    PlaneModel fit_plane(const Eigen::Vector3d& p0,
                         const Eigen::Vector3d& p1,
                         const Eigen::Vector3d& p2) const;

    std::vector<int> find_inliers(const std::vector<Eigen::Vector3d>& pts,
                                   const PlaneModel& plane,
                                   double threshold) const;

    PlaneModel refit_plane(const std::vector<Eigen::Vector3d>& pts,
                            const std::vector<int>& inlier_idx) const;

    /// Esegue un'iterazione RANSAC; rimuove gli inlier da pts.
    /// Restituisce false se non trova un piano con abbastanza inlier.
    bool ransac_once(std::vector<Eigen::Vector3d>& pts, PlaneModel& best_plane);

    bool is_wall_plane(const PlaneModel& plane) const;

    /// Marca i voxel della voxelmap il cui centroide cade su uno dei piani-parete.
    /// Restituisce il numero di voxel marcati.
    int mark_wall_voxels(gtsam_points::DynamicVoxelMapCPU& voxelmap,
                          int nvox,
                          const std::vector<PlaneModel>& wall_planes) const;

    WallFilterConfig  config_;
    mutable std::mt19937 rng_;
};

}  // namespace glim