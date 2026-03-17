
#include <vector>
#include <Eigen/Core>
#include <glim/dynamic_rejection/bounding_box.hpp>


namespace glim {

BoundingBox::BoundingBox(Eigen::Vector3d size, Eigen::Vector3d center, Eigen::Matrix3d rotation)
    : size(size), center(center), rotation(rotation) {
    // Precompute the bounding box limits for faster point-in-box checks
    Eigen::Matrix3d R_inv = rotation.transpose();
    Eigen::Vector3d half_size = size / 2.0;
    bbox_min_limits.clear();
    bbox_max_limits.clear();
    for (int i = 0; i < 8; ++i) {
        Eigen::Vector3d corner(
            (i & 1 ? half_size.x() : -half_size.x()),
            (i & 2 ? half_size.y() : -half_size.y()),
            (i & 4 ? half_size.z() : -half_size.z())
        );
    }
}

BoundingBox::~BoundingBox() = default;

#include <glim/dynamic_rejection/bounding_box.hpp>
#include <cmath>

namespace glim {

BoundingBox::BoundingBox(const Eigen::Vector3d& size,
                         const Eigen::Vector3d& center,
                         const Eigen::Matrix3d& rotation)
    : size(size),
      center(center),
      rotation(rotation)
{
    // Precompute values used in contains()
    R_inv = rotation.transpose();
    half_size = size * 0.5;
}



bool BoundingBox::contains(const Eigen::Vector4d& point) const {
    // Estrarre solo le coordinate xyz
    Eigen::Vector3d p = point.head<3>();

    // Trasformare il punto nel frame locale della bounding box
    Eigen::Vector3d local_p = R_inv * (p - center);

    // Controllo limiti
    return (std::abs(local_p.x()) <= half_size.x() &&
            std::abs(local_p.y()) <= half_size.y() &&
            std::abs(local_p.z()) <= half_size.z());
}

}  // namespace glim
