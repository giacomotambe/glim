#include <glim/dynamic_rejection/dynamic_bounding_box_rejection.hpp>
#include <glim/preprocess/preprocessed_frame.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <spdlog/spdlog.h>

namespace glim {
DynamicBBoxRejection::DynamicBBoxRejection(const std::vector<BoundingBox>& bbox) : bboxes_(bbox) {
    cloud_preprocessor_ = std::make_unique<CloudPreprocessor>();
}

DynamicBBoxRejection::~DynamicBBoxRejection() = default;

PreprocessedFrame::Ptr DynamicBBoxRejection::reject(const PreprocessedFrame::Ptr frame) {
    std::vector<Eigen::Vector4d> filtered_points;
    std::vector<double> filtered_intensities;
    std::vector<double> filtered_times;

    std::vector<Eigen::Vector4d> dynamic_points;
    std::vector<double> dynamic_intensities;
    std::vector<double> dynamic_times;

    for (size_t i = 0; i < frame->points.size(); ++i) {
        const Eigen::Vector4d& point = frame->points[i];
        bool is_dynamic = false;
        for (const auto& bbox : bboxes_) {
            if (!bbox.contains(point)) {
                is_dynamic = true;
                dynamic_points.push_back(point);
                if (frame->intensities) {
                    dynamic_intensities.push_back(frame->intensities->at(i));
                }
                dynamic_times.push_back(frame->times[i]);
                break;
            }
        }
        if (!is_dynamic) {
            filtered_points.push_back(point);
            if (frame->intensities) {
                filtered_intensities.push_back(frame->intensities->at(i));
            }
            filtered_times.push_back(frame->times[i]);

        }
    }

    PreprocessedFrame::Ptr preprocessed(new PreprocessedFrame);
    preprocessed->stamp = frame->stamp;
    preprocessed->scan_end_time = frame->scan_end_time;

    preprocessed->times = filtered_times;
    preprocessed->points = filtered_points;
    if (frame->intensities) {
        preprocessed->intensities = filtered_intensities;
    }

    preprocessed->k_neighbors = params.k_correspondences;
    preprocessed->neighbors = find_neighbors(filtered_points, filtered_points.size(), params.k_correspondences);

    spdlog::trace("preprocessed: {} -> {} points", frame->size(), preprocessed->size());

    if (!dynamic_points.empty()) {
        last_dynamic_frame = std::make_shared<PreprocessedFrame>();
        last_dynamic_frame->stamp = frame->stamp;
        last_dynamic_frame->scan_end_time = frame->scan_end_time;
        last_dynamic_frame->points = dynamic_points;
        last_dynamic_frame->times = dynamic_times;
        if (frame->intensities) {
            last_dynamic_frame->intensities = dynamic_intensities;
        }
    }

    return preprocessed;
}

void DynamicBBoxRejection::insert_bounding_boxes(BoundingBox& bbox) {
    bboxes_.push_back(bbox);
}
}  // namespace glim