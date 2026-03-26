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
    return cluster_bbox_queue_.get_all_and_clear();
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
            
            DynamicClusterExtractor::ClusterMap cluster_map;
 
            // if (cluster_extractor_ && wf.voxelmap) {
            //     cluster_map = cluster_extractor_->cluster_voxels(wf.voxelmap);
            //     spdlog::debug("[dynamic_rejection][async] cluster_map size={}", cluster_map.size());
            // }

            // ------------------------------------------------------------------
            // Step 2: DynamicObjectRejection — score non-wall voxels
            // ------------------------------------------------------------------
            const DynamicRejectionResult dr =
                dynamic_rejection_->reject(wf, frame);
            
            // ------------------------------------------------------------------
            // Enqueue outputs
            // ------------------------------------------------------------------

            // if (cluster_extractor_ && wf.voxelmap && !cluster_map.empty()) {
            //     // Conta il numero di cluster dalla mappa
            //     int num_clusters = 0;
            //     for (int cid : cluster_map)
            //         if (cid >= 0) num_clusters = std::max(num_clusters, cid + 1);
 
            //     if (num_clusters > 0) {
            //         // Raccoglie i punti solo dei cluster dinamici
            //         // (almeno un voxel del cluster ha is_dynamic == true
            //         //  dopo la propagazione)
            //         const int nvox = static_cast<int>(cluster_map.size());
            //         std::vector<bool> cluster_is_dynamic(num_clusters, false);
            //         for (int i = 0; i < nvox; ++i) {
            //             const int cid = cluster_map[i];
            //             if (cid >= 0 && wf.voxelmap->lookup_voxel(i).is_dynamic)
            //                 cluster_is_dynamic[cid] = true;
            //         }
 
            //         // Costruisce point_clusters solo per i cluster dinamici
            //         std::vector<std::vector<Eigen::Vector4d>> dyn_point_clusters(num_clusters);
            //         for (int i = 0; i < nvox; ++i) {
            //             const int cid = cluster_map[i];
            //             if (cid >= 0) {
            //                 const auto& vox = wf.voxelmap->lookup_voxel(i);
            //                 dyn_point_clusters[cid].insert(
            //                     dyn_point_clusters[cid].end(),
            //                     vox.voxel_points.begin(),
            //                     vox.voxel_points.end());
            //             }
            //         }
 
            //         // Rimuove cluster vuoti (statici) e calcola OBB
            //         dyn_point_clusters.erase(
            //             std::remove_if(dyn_point_clusters.begin(), dyn_point_clusters.end(),
            //                 [](const std::vector<Eigen::Vector4d>& c){ return c.empty(); }),
            //             dyn_point_clusters.end());
 
            //         auto bboxes = cluster_extractor_->compute_bounding_boxes(dyn_point_clusters);
 
            //         if (!bboxes.empty()) {
            //             cluster_bbox_queue_.push_back(std::move(bboxes));
            //             spdlog::debug("[dynamic_rejection][async] {} dynamic cluster bboxes enqueued",
            //                           cluster_bbox_queue_.size());
            //         }
            //     }
            // }


            wall_result_queue.push_back(wf);
            output_frame_queue.push_back(dr.static_frame);

            if (dr.dynamic_frame) {
                dynamic_frame_queue.push_back(dr.dynamic_frame);
            }

            // Always enqueue the wall result so the caller can publish wall
            // voxels even when no wall planes were found (num_wall_voxels == 0).

            
            
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