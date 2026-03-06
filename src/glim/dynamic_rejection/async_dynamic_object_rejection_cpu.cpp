#include <glim/dynamic_rejection/async_dynamic_object_rejection.hpp>
#include <spdlog/spdlog.h>

namespace glim {

AsyncDynamicObjectRejection::AsyncDynamicObjectRejection(const std::shared_ptr<DynamicObjectRejectionCPU>& dynamic_rejection)
  : dynamic_rejection(dynamic_rejection) {
  spdlog::info("[dynamic_rejection][async] ctor: dynamic_rejection_ptr={}", static_cast<const void*>(dynamic_rejection.get()));
  kill_switch = false;
  end_of_sequence = false;
  thread = std::thread([this] { run(); });
}

AsyncDynamicObjectRejection::~AsyncDynamicObjectRejection() {
  spdlog::info("[dynamic_rejection][async] dtor begin");
  kill_switch = true;
  join();
  spdlog::info("[dynamic_rejection][async] dtor end");
}

void AsyncDynamicObjectRejection::insert_frame(const glim::PreprocessedFrame::Ptr& frame, glim::EstimationFrame::ConstPtr prev_frame) {
  spdlog::info("[dynamic_rejection][async] insert_frame: frame={} prev_frame={} queue_before={}",
                static_cast<bool>(frame),
                static_cast<bool>(prev_frame),
                input_frame_queue.size());
  input_frame_queue.push_back(std::make_pair(frame, prev_frame));
}

void AsyncDynamicObjectRejection::join() {
  spdlog::info("[dynamic_rejection][async] join begin");
  end_of_sequence = true;
  if (thread.joinable()) {
    thread.join();
  }
  spdlog::info("[dynamic_rejection][async] join end");
}

int AsyncDynamicObjectRejection::workload() const {
  const int size = input_frame_queue.size();
  spdlog::info("[dynamic_rejection][async] workload={}", size);
  return size;
}

std::vector<glim::PreprocessedFrame::Ptr> AsyncDynamicObjectRejection::get_results() {
  auto results = output_frame_queue.get_all_and_clear();
  spdlog::info("[dynamic_rejection][async] get_results: count={}", results.size());
  return results;
}

void AsyncDynamicObjectRejection::run() {
  spdlog::info("[dynamic_rejection][async] run thread started");
  while (!kill_switch) {
    auto frame_pairs = input_frame_queue.get_all_and_clear();

    if (frame_pairs.empty()) {
      if (end_of_sequence) {
        spdlog::info("[dynamic_rejection][async] run thread stopping: end_of_sequence=true");
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    spdlog::info("[dynamic_rejection][async] run batch: size={}", frame_pairs.size());

    for (const auto& frame_pair : frame_pairs) {
      const auto& frame = frame_pair.first;
      const auto& prev_frame = frame_pair.second;

      // Perform dynamic object rejection
      auto processed_frame = dynamic_rejection->dynamic_object_rejection(frame, prev_frame);
      spdlog::info("[dynamic_rejection][async] processed frame: ok={} points={}",
                    static_cast<bool>(processed_frame),
                    processed_frame ? processed_frame->points.size() : 0);

      // Add to output queue
      output_frame_queue.push_back(processed_frame);
    }
  }
  spdlog::info("[dynamic_rejection][async] run thread exited");
}

}  // namespace glim