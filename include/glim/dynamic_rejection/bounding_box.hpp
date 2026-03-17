#pragma once

#include <Eigen/Core>

namespace glim {

class BoundingBox {
public:
    BoundingBox(const Eigen::Vector3d& size,
                const Eigen::Vector3d& center,
                const Eigen::Matrix3d& rotation);

    ~BoundingBox() = default;

    bool contains(const Eigen::Vector4d& point) const;

private:
    Eigen::Vector3d size;
    Eigen::Vector3d center;
    Eigen::Matrix3d rotation;

    // precomputati per velocità
    Eigen::Matrix3d R_inv;
    Eigen::Vector3d half_size;
};

}  // namespace glim