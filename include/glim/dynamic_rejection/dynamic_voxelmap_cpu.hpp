#include <Eigen/Geometry>
#include <Eigen/Core>

#include <gtsam_points/type/gaussian_voxelmap_cpu.hpp>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/gaussian_voxelmap.hpp>
#include <gtsam_points/ann/incremental_voxelmap.hpp>

namespace dynamic_glim {
struct DynamicGaussianVoxel : public gtsam_points::GaussianVoxel
{
    public:
        using Ptr = std::shared_ptr<DynamicGaussianVoxel>;
        using ConstPtr = std::shared_ptr<const DynamicGaussianVoxel>;

        DynamicGaussianVoxel() : gtsam_points::GaussianVoxel(), is_dynamic(false) {point_indices.clear();}
    public:
        bool is_dynamic;
        std::vector<Eigen::Vector4d> voxel_points; 
        std::vector<double> voxel_intensities;
        std::vector<double> voxel_times;

        virtual void add(const Setting& setting, const PointCloud& points, size_t i) override;    
};

class DynamicVoxelMapCPU : public gtsam_points::GaussianVoxelMapCPU , public gtsam_points::IncrementalVoxelMap<DynamicGaussianVoxel> {
    public:
        using Ptr = std::shared_ptr<DynamicVoxelMapCPU>;
        using ConstPtr = std::shared_ptr<const DynamicVoxelMapCPU>;
        DynamicVoxelMapCPU(const double resolution) : gtsam_points::GaussianVoxelMapCPU(resolution) {}

        virtual ~DynamicVoxelMapCPU() override {}

        /**
         * @brief Look up a voxel by its ID
         * @param voxel_id The ID of the voxel to look up
         * @return A constant reference to the DynamicGaussianVoxel
         */
        const DynamicGaussianVoxel& lookup_voxel(int voxel_id) const;

        /**
         * @brief Insert a point cloud frame into the voxel map
         * @param frame The point cloud frame to insert
         */
        void insert(const PointCloud& frame);

}

}