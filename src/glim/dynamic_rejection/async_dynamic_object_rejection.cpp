#include <glim/dynamic_rejection/async_dynamic_object_rejection.hpp>

namespace glim {

AsyncDynamicObjectRejection::AsyncDynamicObjectRejection(const std::shared_ptr<DynamicObjectRejectionCPU>& dynamic_rejection)
  : dynamic_rejection(dynamic_rejection) {
  kill_switch = false;
  end_of_sequence = false;
  thread = std::thread([this] { run(); });
}

AsyncDynamicObjectRejection::~AsyncDynamicObjectRejection() {
  kill_switch = true;
  join();
}

void AsyncDynamicObjectRejection::insert_frame(const glim::PreprocessedFrame::Ptr& frame, glim::EstimationFrame::ConstPtr prev_frame) {
  input_frame_queue.push_back(std::make_pair(frame, prev_frame));
}

void AsyncDynamicObjectRejection::join() {
  end_of_sequence = true;
  if (thread.joinable()) {
    thread.join();
  }
}

int AsyncDynamicObjectRejection::workload() const {
  return input_frame_queue.size();
}

std::vector<glim::PreprocessedFrame::Ptr> AsyncDynamicObjectRejection::get_results() {
  return output_frame_queue.get_all_and_clear();
}

void AsyncDynamicObjectRejection::run() {
  while (!kill_switch) {
    auto frame_pairs = input_frame_queue.get_all_and_clear();

    if (frame_pairs.empty()) {
      if (end_of_sequence) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    for (const auto& frame_pair : frame_pairs) {
      const auto& frame = frame_pair.first;
      const auto& prev_frame = frame_pair.second;

      // Perform dynamic object rejection
      auto processed_frame = dynamic_rejection->dynamic_object_rejection(frame, prev_frame);

      // Add to output queue
      output_frame_queue.push_back(processed_frame);
    }
  }
}

}  // namespace glim