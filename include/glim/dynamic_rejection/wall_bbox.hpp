#pragma once

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <glim/dynamic_rejection/bounding_box.hpp>

namespace glim {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct WallBBoxRegistryConfig {
    /// Soglia IoU [0, 1] sopra la quale una nuova OBB viene fusa con
    /// una esistente invece di essere aggiunta come nuova.
    double overlap_threshold;

    /// Peso [0, 1] della nuova osservazione nella media pesata durante la
    /// fusione.  0.0 → ignora nuova osservazione, 1.0 → sostituisce.
    double merge_weight;

    /// Se true, rimuove dal registro le box che non vengono osservate per
    /// più di max_missed_frames frame consecutivi.
    bool enable_expiry;
    int  max_missed_frames;

    WallBBoxRegistryConfig();
    ~WallBBoxRegistryConfig();    
};

// ---------------------------------------------------------------------------
// WallBBoxRegistry
//
// Mantiene un registro persistente di bounding box di pareti.
// Ad ogni chiamata a update() le OBB del frame corrente vengono confrontate
// con quelle già salvate tramite overlap IoU (Intersection over Union).
//
//   - Se IoU >= overlap_threshold  → la box esistente viene aggiornata
//     (media pesata tra quella salvata e quella nuova, SLERP per rotazione).
//   - Se IoU <  overlap_threshold  → la box viene aggiunta come nuova parete.
//
// L'overlap è calcolato con la Separating Axis Theorem (SAT) su OBB 3D,
// che gestisce correttamente box ruotate a differenza delle AABB.
// ---------------------------------------------------------------------------

class WallBBoxRegistry {
public:
    using Ptr      = std::shared_ptr<WallBBoxRegistry>;
    using ConstPtr = std::shared_ptr<const WallBBoxRegistry>;

    explicit WallBBoxRegistry(
        const WallBBoxRegistryConfig& config = WallBBoxRegistryConfig{});

    /// Confronta le nuove OBB con il registro e aggiorna (fonde o aggiunge).
    void update(const std::vector<BoundingBox>& new_bboxes);

    /// Registro corrente di bounding box di pareti (lettura).
    const std::vector<BoundingBox>& bboxes() const { return registry_; }

    /// Svuota il registro.
    // void clear();

private:
    // -----------------------------------------------------------------------
    // OBB helpers (Separating Axis Theorem)
    // -----------------------------------------------------------------------

    struct OBB {
        Eigen::Vector3d center;
        Eigen::Matrix3d axes;         ///< colonne = assi locali normalizzati
        Eigen::Vector3d half_extents;
    };

    static OBB    to_obb(const BoundingBox& bbox);
    static double project_obb(const OBB& obb, const Eigen::Vector3d& axis);
    static double interval_overlap(double c1, double r1, double c2, double r2);
    static double intersection_volume(const OBB& a, const OBB& b);
    static double iou(const BoundingBox& a, const BoundingBox& b);

    // -----------------------------------------------------------------------
    // Merge
    // -----------------------------------------------------------------------

    static BoundingBox merge(const BoundingBox& existing,
                             const BoundingBox& incoming,
                             double weight);

private:
    WallBBoxRegistryConfig   config_;
    std::vector<BoundingBox> registry_;
    std::vector<int>         missed_frames_;  ///< contatore per expiry
};

}  // namespace glim