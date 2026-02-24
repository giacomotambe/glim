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

        DynamicGaussianVoxel() : gtsam_points::GaussianVoxel(), is_dynamic(false) {}
    public:
        bool is_dynamic;
        
};

class DynamicVoxelMapCPU : public gtsam_points::GaussianVoxelMapCPU {
    public:
        using Ptr = std::shared_ptr<DynamicVoxelMapCPU>;
        using ConstPtr = std::shared_ptr<const DynamicVoxelMapCPU>;
        DynamicVoxelMapCPU(const double resolution) : gtsam_points::GaussianVoxelMapCPU(resolution) {}

        virtual ~DynamicVoxelMapCPU() override {}

        virtual void insert(const gtsam_points::PointCloud::ConstPtr& cloud) override {
            gtsam_points::GaussianVoxelMapCPU::insert(cloud);
            // After inserting the point cloud, we need to initialize the is_dynamic flag for each voxel
            for (size_t i = 0; i < this->num_voxel(); ++i) {
                auto voxel = std::dynamic_pointer_cast<DynamicGaussianVoxel>(this->lookup_voxel(i));
                if (voxel) {
                    voxel->is_dynamic = false; // Initialize all voxels as static
                }
            }
        }
        
        void get_dynamic_voxels_indices(std::vector<int>& dynamic_voxels_indices) const {
            dynamic_voxels_indices.clear();
            for (size_t i = 0; i < this->num_voxel(); ++i) {
                auto voxel = std::dynamic_pointer_cast<DynamicGaussianVoxel>(this->lookup_voxel(i));
                if (voxel && voxel->is_dynamic) {
                    dynamic_voxels_indices.push_back(i);
                }
            }
        }

        void set_dynamic_voxel(const int voxel_index) {
            auto voxel = std::dynamic_pointer_cast<DynamicGaussianVoxel>(this->lookup_voxel(voxel_index));
            if (voxel) {
                voxel->is_dynamic = true;
            }
        }

        

}

}