#include <algorithm>
#include <chrono>

#include <spdlog/spdlog.h>

#include <glim/viewer/dynamic_rejection_viewer.hpp>
#include <glim/odometry/callbacks.hpp>
#include <glim/util/config.hpp>
#include <glim/util/logging.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

namespace glim {

DynamicRejectionViewer::DynamicRejectionViewer()
    : callback_id(-1),
      viewer_started(false),
      request_to_terminate(false),
      kill_switch(false),
      viewer_size(1280, 720),
      point_size(0.025),
      point_size_metric(true),
      point_shape_circle(true),
      logger(create_module_logger("dynamic_rejection_viewer")) {
  Config config(GlobalConfig::get_config_path("config_viewer"));

  viewer_size = Eigen::Vector2i(
    config.param("standard_viewer", "viewer_width", 1280),
    config.param("standard_viewer", "viewer_height", 720));
  point_size = config.param("standard_viewer", "point_size", 0.025);
  point_size_metric = config.param("standard_viewer", "point_size_metric", true);
  point_shape_circle = config.param("standard_viewer", "point_shape_circle", true);

  callback_id = OdometryEstimationCallbacks::on_insert_frame.add([this](const PreprocessedFrame::Ptr& frame) {
    update_frame(frame);
  });

  thread = std::thread([this] { viewer_loop(); });

  const auto t1 = std::chrono::high_resolution_clock::now();
  while (!viewer_started && std::chrono::high_resolution_clock::now() - t1 < std::chrono::seconds(1)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  if (!viewer_started) {
    logger->critical("Timeout waiting for dynamic rejection viewer to start");
  }
}

DynamicRejectionViewer::~DynamicRejectionViewer() {
  OdometryEstimationCallbacks::on_insert_frame.remove(callback_id);
  kill_switch = true;

  if (thread.joinable()) {
    thread.join();
  }
}

bool DynamicRejectionViewer::ok() const {
  return !request_to_terminate;
}

void DynamicRejectionViewer::invoke(const std::function<void()>& task) {
  if (kill_switch) {
    return;
  }

  std::lock_guard<std::mutex> lock(invoke_queue_mutex);
  invoke_queue.push_back(task);
}

void DynamicRejectionViewer::update_frame(const PreprocessedFrame::Ptr& frame) {
  if (!frame) {
    return;
  }

  invoke([frame] {
    auto viewer = guik::LightViewer::instance();

    if (frame->points.empty()) {
      viewer->remove_drawable("dynamic_rejection_points");
      return;
    }

    auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(frame->points.data(), frame->points.size());

    if (frame->intensities.size() == frame->points.size()) {
      const auto max_it = std::max_element(frame->intensities.begin(), frame->intensities.end());
      const double max_intensity = max_it != frame->intensities.end() ? *max_it : 0.0;
      if (max_intensity > 0.0) {
        cloud_buffer->add_intensity(glk::COLORMAP::TURBO, frame->intensities.data(), frame->points.size(), 1.0 / max_intensity);
      }
    }

    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto& pt : frame->points) {
      center += pt.head<3>().cast<float>();
    }
    center /= static_cast<float>(frame->points.size());
    viewer->lookat(center);

    auto shader = guik::FlatColor(1.0f, 0.5f, 0.0f, 1.0f);
    if (frame->intensities.size() == frame->points.size()) {
      shader.set_color_mode(guik::ColorMode::VERTEX_COLORMAP);
    }

    viewer->update_drawable("dynamic_rejection_points", cloud_buffer, shader.add("point_scale", 2.0f));
  });
}

void DynamicRejectionViewer::viewer_loop() {
  auto viewer = guik::LightViewer::instance(viewer_size);
  viewer_started = true;

  viewer->enable_vsync();
  viewer->shader_setting().set_point_size(point_size);

  if (point_size_metric) {
    viewer->shader_setting().set_point_scale_metric();
  }

  if (point_shape_circle) {
    viewer->shader_setting().set_point_shape_circle();
  }

  while (!kill_switch) {
    if (!viewer->spin_once()) {
      request_to_terminate = true;
    }

    std::vector<std::function<void()>> tasks;
    {
      std::lock_guard<std::mutex> lock(invoke_queue_mutex);
      tasks.swap(invoke_queue);
    }

    for (const auto& task : tasks) {
      task();
    }
  }

  viewer->remove_drawable("dynamic_rejection_points");
  guik::LightViewer::destroy();
}

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::DynamicRejectionViewer();
}