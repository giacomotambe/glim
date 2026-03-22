#include <glim/dynamic_rejection/async_dynamic_object_rejection.hpp>
#include <spdlog/spdlog.h>

namespace glim {

AsyncDynamicObjectRejection::AsyncDynamicObjectRejection(
    const std::shared_ptr<DynamicObjectRejectionCPU>& dynamic_rejection,
    const std::shared_ptr<WallFilter>&                wall_filter,
    const std::shared_ptr<DynamicClusterExtractor>&   cluster_extractor)
  : dynamic_rejection_(dynamic_rejection),
    wall_filter_(wall_filter),
    cluster_extractor_(cluster_extractor)
{
    spdlog::debug("[dynamic_rejection][async] ctor");
    kill_switch     = false;
    end_of_sequence = false;
    thread          = std::thread([this] { run(); });
}

AsyncDynamicObjectRejection::~AsyncDynamicObjectRejection() {
    spdlog::debug("[dynamic_rejection][async] dtor begin");
    kill_switch = true;
    join();
    spdlog::debug("[dynamic_rejection][async] dtor end");
}

void AsyncDynamicObjectRejection::insert_frame(const glim::PreprocessedFrame::Ptr& frame) {
    spdlog::debug("[dynamic_rejection][async] insert_frame: queue_size={}",
                  input_frame_queue.size());
    input_frame_queue.push_back(frame);
}

void AsyncDynamicObjectRejection::join() {
    spdlog::debug("[dynamic_rejection][async] join begin");
    end_of_sequence = true;
    if (thread.joinable()) thread.join();
    spdlog::debug("[dynamic_rejection][async] join end");
}

int AsyncDynamicObjectRejection::workload() const {
    return static_cast<int>(input_frame_queue.size());
}

std::vector<glim::PreprocessedFrame::Ptr> AsyncDynamicObjectRejection::get_results() {
    auto results = output_frame_queue.get_all_and_clear();
    spdlog::debug("[dynamic_rejection][async] get_results: count={}", results.size());
    return results;
}

std::vector<glim::PreprocessedFrame::Ptr> AsyncDynamicObjectRejection::get_dynamic_results() {
    return dynamic_frame_queue.get_all_and_clear();
}

std::vector<WallFilterResult> AsyncDynamicObjectRejection::get_wall_results() {
    return wall_result_queue.get_all_and_clear();
}

std::vector<std::vector<BoundingBox>> AsyncDynamicObjectRejection::get_cluster_bbox_results() {
    return cluster_bbox_queue.get_all_and_clear();
}

void AsyncDynamicObjectRejection::run() {
    spdlog::debug("[dynamic_rejection][async] thread started");

    while (!kill_switch) {
        auto frames = input_frame_queue.get_all_and_clear();

        if (frames.empty()) {
            if (end_of_sequence) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        spdlog::debug("[dynamic_rejection][async] processing batch of {} frames", frames.size());

        for (const auto& frame : frames) {
            // ------------------------------------------------------------------
            // Step 1: WallFilter — voxelize + mark wall voxels
            // ------------------------------------------------------------------
            const WallFilterResult wf = wall_filter_->filter(*frame);
            const std::vector<BoundingBox> bboxes = cluster_extractor_->extract_clusters(wf.voxelmap);
            spdlog::debug("[dynamic_rejection][async] found {} cluster bounding boxes", bboxes.size());

            // ------------------------------------------------------------------
            // Step 2: DynamicObjectRejection — score non-wall voxels
            // ------------------------------------------------------------------
            const DynamicRejectionResult dr =
                dynamic_rejection_->reject(wf, frame);
            
            // ------------------------------------------------------------------
            // Enqueue outputs
            // ------------------------------------------------------------------
            output_frame_queue.push_back(dr.static_frame);

            if (dr.dynamic_frame) {
                dynamic_frame_queue.push_back(dr.dynamic_frame);
            }

            // Always enqueue the wall result so the caller can publish wall
            // voxels even when no wall planes were found (num_wall_voxels == 0).

            if (!bboxes.empty()) {
                cluster_bbox_queue.push_back(std::move(bboxes));
                spdlog::debug("[dynamic_rejection][async] {} dynamic cluster bboxes enqueued",
                                cluster_bbox_queue.size());
            }
            wall_result_queue.push_back(wf);
            
            spdlog::debug("[dynamic_rejection][async] frame done: "
                          "wall_voxels={}/{} static_pts={} dynamic_pts={}",
                wf.num_wall_voxels,
                wf.num_total_voxels,
                dr.static_frame  ? dr.static_frame->points.size()  : 0,
                dr.dynamic_frame ? dr.dynamic_frame->points.size() : 0);
        }
    }

    spdlog::debug("[dynamic_rejection][async] thread exited");
}

}  // namespace glim