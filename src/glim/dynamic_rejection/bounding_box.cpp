
#include <vector>
#include <Eigen/Core>
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


void BoundingBox::transform(const Eigen::Isometry3d& T) {
    // Aggiorna centro e rotazione
    center = T * center;
    rotation = T.linear() * rotation;

    // Aggiorna la matrice inversa per contains()
    R_inv = rotation.transpose();
}

}  // namespace glim