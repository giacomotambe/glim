#include <glim/dynamic_rejection/async_dynamic_object_rejection.hpp>
#include <spdlog/spdlog.h>

namespace glim {

AsyncDynamicObjectRejection::AsyncDynamicObjectRejection(const std::shared_ptr<DynamicObjectRejectionCPU>& dynamic_rejection)
  : dynamic_rejection(dynamic_rejection) {
  spdlog::debug("[dynamic_rejection][async] ctor: dynamic_rejection_ptr={}", static_cast<const void*>(dynamic_rejection.get()));
  kill_switch = false;
  end_of_sequence = false;
  thread = std::thread([this] { run(); });
}

AsyncDynamicObjectRejection::~AsyncDynamicObjectRejection() {
  spdlog::debug("[dynamic_rejection][async] dtor begin");
  kill_switch = true;
  join();
  spdlog::debug("[dynamic_rejection][async] dtor end");
}

void AsyncDynamicObjectRejection::insert_frame(const glim::PreprocessedFrame::Ptr& frame) {
  spdlog::debug("[dynamic_rejection][async] insert_frame: frame={} queue_before={}",
                static_cast<bool>(frame),
                input_frame_queue.size());
  input_frame_queue.push_back(frame);
}

void AsyncDynamicObjectRejection::join() {
  spdlog::debug("[dynamic_rejection][async] join begin");
  end_of_sequence = true;
  if (thread.joinable()) {
    thread.join();
  }
  spdlog::debug("[dynamic_rejection][async] join end");
}

int AsyncDynamicObjectRejection::workload() const {
  const int size = input_frame_queue.size();
  spdlog::debug("[dynamic_rejection][async] workload={}", size);
  return size;
}

std::vector<glim::PreprocessedFrame::Ptr> AsyncDynamicObjectRejection::get_results() {
  auto results = output_frame_queue.get_all_and_clear();
  spdlog::debug("[dynamic_rejection][async] get_results: count={}", results.size());
  return results;
}

std::vector<glim::PreprocessedFrame::Ptr> AsyncDynamicObjectRejection::get_dynamic_results() {
  return dynamic_frame_queue.get_all_and_clear();
}

void AsyncDynamicObjectRejection::run() {
  spdlog::debug("[dynamic_rejection][async] run thread started");
  while (!kill_switch) {
    auto frames = input_frame_queue.get_all_and_clear();

    if (frames.empty()) {
      if (end_of_sequence) {
        spdlog::debug("[dynamic_rejection][async] run thread stopping: end_of_sequence=true");
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    spdlog::debug("[dynamic_rejection][async] run batch: size={}", frames.size());

    for (const auto& frame : frames) {

      // Perform dynamic object rejection
      auto processed_frame = dynamic_rejection->dynamic_object_rejection(frame);
      spdlog::debug("[dynamic_rejection][async] processed frame: ok={} points={}",
                    static_cast<bool>(processed_frame),
                    processed_frame ? processed_frame->points.size() : 0);

      // Add to output queue
      output_frame_queue.push_back(processed_frame);

      // Add dynamic-only frame to separate queue
      auto dyn_frame = dynamic_rejection->get_last_dynamic_frame();
      if (dyn_frame) {
        dynamic_frame_queue.push_back(dyn_frame);
      }
    }
  }
  spdlog::debug("[dynamic_rejection][async] run thread exited");
}

}  // namespace glim