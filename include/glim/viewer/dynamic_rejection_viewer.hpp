#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <Eigen/Core>

#include <glim/preprocess/preprocessed_frame.hpp>
#include <glim/util/extension_module.hpp>

namespace spdlog {
class logger;
}

namespace glim {

class DynamicRejectionViewer : public ExtensionModule {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DynamicRejectionViewer();
  ~DynamicRejectionViewer() override;

  bool ok() const override;

private:
  void invoke(const std::function<void()>& task);
  void update_frame(const PreprocessedFrame::Ptr& frame);
  void viewer_loop();

private:
  int callback_id;

  std::atomic_bool viewer_started;
  std::atomic_bool request_to_terminate;
  std::atomic_bool kill_switch;
  std::thread thread;

  std::mutex invoke_queue_mutex;
  std::vector<std::function<void()>> invoke_queue;

  Eigen::Vector2i viewer_size;
  double point_size;
  bool point_size_metric;
  bool point_shape_circle;

  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim