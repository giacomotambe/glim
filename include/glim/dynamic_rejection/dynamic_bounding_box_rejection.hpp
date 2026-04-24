#include <memory>
#include <vector>
#include <Eigen/Geometry>
#include <glim/dynamic_rejection/bounding_box.hpp>
#include <glim/dynamic_rejection/transformation_kalman_filter.hpp>
#include <glim/preprocess/preprocessed_frame.hpp>

namespace glim {
class DynamicBBoxRejection {
public:
    using Ptr = std::shared_ptr<DynamicBBoxRejection>;
    using ConstPtr = std::shared_ptr<const DynamicBBoxRejection>;

    DynamicBBoxRejection(const std::vector<BoundingBox>& bbox,
                         const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter = nullptr);
    explicit DynamicBBoxRejection(const std::shared_ptr<PoseKalmanFilter>& pose_kalman_filter = nullptr);
    ~DynamicBBoxRejection();

    /**
     * @brief Classify points as dynamic if they are outside the bounding box and return a new estimation frame without points classified as dynamic.
     * @param frame Current preprocessed frame
     * @return New preprocessed frame with dynamic points removed
     */
    PreprocessedFrame::Ptr reject(const PreprocessedFrame::Ptr frame);  
    void insert_bounding_boxes(BoundingBox& bbox);
    void clear_bounding_boxes() { bboxes_.clear(); }
    void set_bounding_boxes(const std::vector<BoundingBox>& bboxes) { bboxes_ = bboxes; }
    PreprocessedFrame::Ptr get_last_dynamic_frame() const { return last_dynamic_frame; }
private:
    std::vector<int> find_neighbors(const Eigen::Vector4d* points, const int num_points, const int k) const;
private:
    std::vector<BoundingBox> bboxes_;
    PreprocessedFrame::Ptr last_dynamic_frame = nullptr;
    std::shared_ptr<PoseKalmanFilter> pose_kalman_filter_;
};  

}  // namespace glim    