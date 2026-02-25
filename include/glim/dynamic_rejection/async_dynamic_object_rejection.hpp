#pragma once

#include <atomic>
#include <thread>
#include <memory>
#include <glim/dynamic_rejection/dynamic_object_rejection_cpu.hpp>
#include <glim/util/concurrent_vector.hpp>
#include <glim/preprocess/preprocessed_frame.hpp>
#include <glim/odometry/estimation_frame.hpp>

namespace glim {

/**
 * @brief Async wrapper for DynamicObjectRejectionCPU to run dynamic object rejection asynchronously
 * @note  All the exposed public methods are thread-safe
 */
class AsyncDynamicObjectRejection {
public:
  /**
   * @brief Construct a new Async Dynamic Object Rejection object
   * @param dynamic_rejection Dynamic object rejection object to wrap
   */
  AsyncDynamicObjectRejection(const std::shared_ptr<DynamicObjectRejectionCPU>& dynamic_rejection);

  /**
   * @brief Destroy the Async Dynamic Object Rejection object
   */
  ~AsyncDynamicObjectRejection();

  /**
   * @brief Insert a frame pair for dynamic object rejection
   * @param frame Current preprocessed frame
   * @param prev_frame Previous estimation frame
   */
  void insert_frame(const glim::PreprocessedFrame::Ptr& frame, glim::EstimationFrame::ConstPtr prev_frame);

  /**
   * @brief Wait for the dynamic rejection thread
   */
  void join();

  /**
   * @brief Number of data in the input queue (for load control)
   * @return Input queue size
   */
  int workload() const;

  /**
   * @brief Get the processed frames without dynamic objects
   * @return Processed frames
   */
  std::vector<glim::PreprocessedFrame::Ptr> get_results();

private:
  void run();

private:
  std::atomic_bool kill_switch;      // Flag to stop the thread immediately (Hard kill switch)
  std::atomic_bool end_of_sequence;  // Flag to stop the thread when the input queues become empty (Soft kill switch)
  std::thread thread;

  // Input queue: pairs of (current_frame, prev_frame)
  ConcurrentVector<std::pair<glim::PreprocessedFrame::Ptr, glim::EstimationFrame::ConstPtr>> input_frame_queue;

  // Output queue
  ConcurrentVector<glim::PreprocessedFrame::Ptr> output_frame_queue;

  std::shared_ptr<DynamicObjectRejectionCPU> dynamic_rejection;
};

}  // namespace glim