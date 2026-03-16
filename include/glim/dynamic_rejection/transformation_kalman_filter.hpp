#pragma once

#include <mutex>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim {

struct ImuMeasurement {
  Eigen::Vector3d acc;
  Eigen::Vector3d gyro;
  double dt;
};

/**
 * @brief Error-State Kalman Filter that estimates the absolute pose
 *        T_world_imu by fusing IMU predictions with SLAM updates.
 *
 * Between SLAM updates the filter integrates IMU measurements to accumulate
 * the incremental motion (position delta + rotation delta + velocity).
 *
 * When a new SLAM pose arrives, update() computes the SLAM-observed delta,
 * fuses it with the IMU prediction, and returns the filtered absolute pose
 * T_world_imu.
 *
 * Nominal state (relative to previous frame):
 *   Δp (3)  – position delta
 *   Δv (3)  – velocity at current time
 *   Δq (4)  – rotation delta (quaternion)
 *
 * Error state (9-dim): δp (3), δv (3), δθ (3)
 */
class PoseKalmanFilter {
public:
  PoseKalmanFilter();

  /// Return the current accumulated relative transform (T_{k-1 -> k})
  Eigen::Isometry3d getDeltaPose() const;


  /// Return the current velocity estimate
  Eigen::Vector3d getVelocity() const;

  /// IMU prediction: accumulate relative motion
  void predict(const ImuMeasurement& imu);

  /**
   * @brief SLAM update: fuse the SLAM-observed absolute pose.
   *
   * Internally computes the SLAM-measured delta from the previous SLAM pose,
   * fuses with IMU-integrated delta, resets for next interval, and returns
   * the filtered absolute pose T_world_imu.
   *
   * @param T_world_imu  Absolute SLAM pose at current time
   * @return Filtered absolute pose T_world_imu
   */
  Eigen::Isometry3d update(const Eigen::Isometry3d& T_world_imu);

private:
  static Eigen::Matrix3d skew(const Eigen::Vector3d& v);

  // Nominal relative state (accumulated since last reset)
  Eigen::Vector3d delta_position_;
  Eigen::Vector3d velocity_;
  Eigen::Quaterniond delta_orientation_;

  // Orientation at the start of the current interval (for rotating acc to local-level frame)
  Eigen::Quaterniond orientation_at_reset_;

  // Previous SLAM pose (to compute SLAM-observed delta)
  Eigen::Isometry3d last_slam_pose_;
  bool has_last_slam_pose_;

  // Error-state covariance (9×9: δp, δv, δθ)
  Eigen::Matrix<double, 9, 9> P_;
  Eigen::Matrix<double, 9, 9> Q_;   // process noise
  Eigen::Matrix<double, 6, 6> R_;   // measurement noise (pos 3 + rot 3)

  Eigen::Vector3d gravity_;

  mutable std::mutex mutex_;  // protects all state from concurrent access
};

}  // namespace glim
