#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim {

class BoundingBox {
public:
    BoundingBox(const Eigen::Vector3d& size,
                const Eigen::Vector3d& center,
                const Eigen::Matrix3d& rotation);

    ~BoundingBox() = default;

    bool contains(const Eigen::Vector4d& point) const;
    void transform(const Eigen::Isometry3d& T);
    // -----------------------------------------------------------------------
    // Getters (needed by WallBBoxRegistry for IoU / merge operations)
    // -----------------------------------------------------------------------
    const Eigen::Vector3d& get_size()     const { return size; }
    const Eigen::Vector3d& get_center()   const { return center; }
    const Eigen::Matrix3d& get_rotation() const { return rotation; }

private:
    Eigen::Vector3d size;
    Eigen::Vector3d center;
    Eigen::Matrix3d rotation;

    // Precomputed for contains()
    Eigen::Matrix3d R_inv;
    Eigen::Vector3d half_size;
};

}  // namespace glim