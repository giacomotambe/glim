#include <vector>
#include <glim/preprocess/preprocessed_frame.hpp>
#include <glim/odometry/estimation_frame.hpp>
#include <gtsam_points/types/gaussian_voxelmap.hpp>


namespace glim {
struct DynamicObjectRejectionParamsCPU {
    public: 
        DynamicObjectRejectionParamsCPU();
        ~DynamicObjectRejectionParamsCPU();
    
    public:
        double mean_difference_threshold; ///< Threshold for mean difference to classify a point as dynamic
        double covariance_error_threshold; ///< Threshold for variance difference to classify a point as dynamic
        int points_number_difference_threshold; ///< Threshold for the number of points difference to classify a point as dynamic
        double voxel_resolution; 
        double voxel_resolution_max; 
        double voxel_resolution_dmin;  
        double voxel_resolution_dmax;
        int voxelmap_levels;
        double voxelmap_scaling_factor;

};


class DynamicObjectRejectionCPU {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Constructor
     */
  DynamicObjectRejectionCPU(const DynamicObjectRejectionParamsCPU& params = DynamicObjectRejectionParamsCPU());

    /**
     * @brief Recognize dynamic objects in the current frame by comparing it with the previous frame and return a new estimation frame without points classified as dynamic.
     * @param frame Current preprocessed frame
     * @param prev_frame Previous estimation frame
     * @return PreprocessedFrame::Ptr New preprocessed frame without dynamic points
    */
  PreprocessedFrame::Ptr dynamic_object_rejection(const PreprocessedFrame::Ptr frame, EstimationFrame::ConstPtr prev_frame);

    /**
     * @brief Get the indices of points classified as dynamic
     * @return std::vector<int> Indices of dynamic points
     */
  std::vector<int> get_dynamic_points_indices() const { return dynamic_voxels_indices; }
private:
  // Voxelize the input frame and return a vector of GaussianVoxelMaps at different resolutions
  std::vector<gtsam_points::GaussianVoxelMap::Ptr> voxelize(const PreprocessedFrame::Ptr& frame, const Eigen::Isometry3d& T_imu_lidar, CloudCovarianceEstimation& covariance_estimation);

  // Add odometry information to the voxelmaps (e.g., by transforming them according to the estimated pose) and return the updated voxelmaps
   std::vector<gtsam_points::GaussianVoxelMap::Ptr> add_odometry(const std::vector<gtsam_points::GaussianVoxelMap::Ptr>& voxelmaps, const Eigen::Isometry3d& T_world_imu);

private:
    DynamicObjectRejectionParamsCPU params_;
    std::vector<int> dynamic_voxels_indices;
    std::vector<std::vector<bool>> is_dynamic_voxel; // This vector will store whether each voxel is classified as dynamic or not 
};

}  // namespace glim
