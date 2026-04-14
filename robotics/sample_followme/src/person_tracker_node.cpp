// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "follow_me/person_tracker_node.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <qrb_ros_people_reid/srv/compute_similarity.hpp>
#include <qrb_ros_people_reid/srv/extract_feature.hpp>

namespace follow_me
{

PersonTrackerNode::PersonTrackerNode() : Node("person_tracker_node"), has_target_(false)
{
  // Declare parameters
  declareParameters();

  // Load parameters
  loadParameters();

  // Initialize modules that do not require shared_from_this()
  template_pool_ = std::make_unique<TemplatePool>(
      template_update_threshold_, match_threshold_, drift_threshold_);
  pid_controller_ = std::make_unique<DualPIDController>(linear_kp_, linear_ki_, linear_kd_,
      angular_kp_, angular_ki_, angular_kd_, max_linear_speed_, max_angular_speed_);
  depth_processor_ = std::make_unique<DepthProcessor>();
  state_machine_ = std::make_unique<StateMachine>();

  extract_feature_client_ =
      this->create_client<qrb_ros_people_reid::srv::ExtractFeature>("/extract_feature");
  compute_similarity_client_ =
      this->create_client<qrb_ros_people_reid::srv::ComputeSimilarity>("/compute_similarity");

  rgb_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
      this, "/camera/color/image_raw");
  depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
      this, "/camera/depth/image_raw");
  detection_sub_ =
      std::make_shared<message_filters::Subscriber<vision_msgs::msg::Detection2DArray>>(
          this, "/yolo_detect_result");

  RCLCPP_INFO(this->get_logger(), "Subscribers created for RGB, Depth, and Detection topics");

  // Create synchronizer with larger queue and longer time window
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(100), *rgb_sub_, *depth_sub_, *detection_sub_);
  sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.01));
  sync_->registerCallback(std::bind(&PersonTrackerNode::syncCallback, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3));

  RCLCPP_INFO(this->get_logger(), "Synchronizer registered with queue_size=20, max_interval=0.5s");

  // Subscribe to camera info for FOV
  camera_info_sub_ =
      this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera/color/camera_info", 10,
          std::bind(&PersonTrackerNode::cameraInfoCallback, this, std::placeholders::_1));

  // Create publishers
  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
  vis_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/target_visualization", 10);

  // Create state control service
  state_control_service_ = this->create_service<follow_me::srv::StateControl>(
      "/follow_me/state_control", std::bind(&PersonTrackerNode::stateControlCallback, this,
                                      std::placeholders::_1, std::placeholders::_2));

  processing_thread_ = std::thread(&PersonTrackerNode::processingLoop, this);

  RCLCPP_INFO(this->get_logger(), "Person Tracker Node initialized");
  RCLCPP_INFO(this->get_logger(), "Detection image size: %d x %d", detection_image_width_,
      detection_image_height_);
  RCLCPP_INFO(this->get_logger(), "Target distance: %.2f m", target_distance_);
  RCLCPP_INFO(this->get_logger(), "Init distance range: [%.2f, %.2f] m", min_init_distance_,
      max_init_distance_);
  RCLCPP_INFO(this->get_logger(), "Template update threshold: %.2f", template_update_threshold_);
  RCLCPP_INFO(this->get_logger(), "Match threshold: %.2f", match_threshold_);
  RCLCPP_INFO(this->get_logger(), "Drift threshold: %.2f", drift_threshold_);
  RCLCPP_INFO(this->get_logger(), "Max processing time: %d ms", max_processing_time_ms_);
}

PersonTrackerNode::~PersonTrackerNode()
{
  // Stop processing thread
  process_thread_running_ = false;
  data_cv_.notify_one();

  if (processing_thread_.joinable()) {
    processing_thread_.join();
  }

  RCLCPP_INFO(this->get_logger(), "Processing thread stopped");
}

void PersonTrackerNode::declareParameters()
{
  // Detection parameters
  this->declare_parameter("detection_image_width", 640);
  this->declare_parameter("detection_image_height", 640);

  // Tracking parameters
  this->declare_parameter("target_distance", 2.0);
  this->declare_parameter("min_init_distance", 1.5);
  this->declare_parameter("max_init_distance", 3.0);
  this->declare_parameter("person_class_id", 0);
  this->declare_parameter("template_update_threshold", 0.1);
  this->declare_parameter("match_threshold", 0.25);
  this->declare_parameter("drift_threshold", 0.5);
  this->declare_parameter("max_processing_time_ms", 200);
  this->declare_parameter("debug_mode", true);

  // PID parameters
  this->declare_parameter("linear_kp", 0.5);
  this->declare_parameter("linear_ki", 0.0);
  this->declare_parameter("linear_kd", 0.1);
  this->declare_parameter("angular_kp", 2.0);
  this->declare_parameter("angular_ki", 0.0);
  this->declare_parameter("angular_kd", 0.2);
  this->declare_parameter("max_linear_speed", 0.5);
  this->declare_parameter("max_angular_speed", 0.5);
}

void PersonTrackerNode::loadParameters()
{
  detection_image_width_ = this->get_parameter("detection_image_width").as_int();
  detection_image_height_ = this->get_parameter("detection_image_height").as_int();

  target_distance_ = this->get_parameter("target_distance").as_double();
  min_init_distance_ = this->get_parameter("min_init_distance").as_double();
  max_init_distance_ = this->get_parameter("max_init_distance").as_double();
  person_class_id_ = this->get_parameter("person_class_id").as_int();
  template_update_threshold_ = this->get_parameter("template_update_threshold").as_double();
  match_threshold_ = this->get_parameter("match_threshold").as_double();
  drift_threshold_ = this->get_parameter("drift_threshold").as_double();
  max_processing_time_ms_ = this->get_parameter("max_processing_time_ms").as_int();
  debug_mode_ = this->get_parameter("debug_mode").as_bool();

  linear_kp_ = this->get_parameter("linear_kp").as_double();
  linear_ki_ = this->get_parameter("linear_ki").as_double();
  linear_kd_ = this->get_parameter("linear_kd").as_double();
  angular_kp_ = this->get_parameter("angular_kp").as_double();
  angular_ki_ = this->get_parameter("angular_ki").as_double();
  angular_kd_ = this->get_parameter("angular_kd").as_double();
  max_linear_speed_ = this->get_parameter("max_linear_speed").as_double();
  max_angular_speed_ = this->get_parameter("max_angular_speed").as_double();
}

void PersonTrackerNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  if (!camera_info_received_) {
    float fx = msg->k[0];
    float width = msg->width;
    fov_horizontal_ = 2.0f * std::atan(width / (2.0f * fx));
    camera_info_received_ = true;
    RCLCPP_INFO(this->get_logger(), "Camera FOV: %.1f degrees", fov_horizontal_ * 180.0 / M_PI);
  }
}

void PersonTrackerNode::syncCallback(const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr & detection_msg)
{
  if (!state_machine_->isRunning()) {
    return;
  }

  try {
    cv_bridge::CvImagePtr cv_rgb = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
    cv_bridge::CvImagePtr cv_depth = cv_bridge::toCvCopy(depth_msg);

    // Store RGB and depth first so extractPersonDetections can use them for coordinate conversion
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      latest_rgb_ = cv_rgb->image.clone();
      latest_depth_ = cv_depth->image.clone();
    }

    // Now extract detections with coordinate conversion
    auto detections = extractPersonDetections(detection_msg);
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      latest_detections_ = detections;
      has_new_data_ = true;
    }

    data_cv_.notify_one();

  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Exception in syncCallback: %s", e.what());
  }
}

void PersonTrackerNode::stateControlCallback(
    const std::shared_ptr<follow_me::srv::StateControl::Request> request,
    std::shared_ptr<follow_me::srv::StateControl::Response> response)
{
  bool success = false;
  std::string message;

  switch (request->set_state) {
    case follow_me::srv::StateControl::Request::STATE_START:
      success = state_machine_->start();
      message = success ? "Started successfully" : "Failed to start";
      RCLCPP_INFO(this->get_logger(), "State control: START - %s", message.c_str());
      break;

    case follow_me::srv::StateControl::Request::STATE_PAUSE:
      success = state_machine_->pause();
      message = success ? "Paused successfully" : "Failed to pause";
      if (success) {
        stopRobot();
      }
      RCLCPP_INFO(this->get_logger(), "State control: PAUSE - %s", message.c_str());
      break;

    case follow_me::srv::StateControl::Request::STATE_RESUME:
      success = state_machine_->resume();
      message = success ? "Resumed successfully" : "Failed to resume";
      RCLCPP_INFO(this->get_logger(), "State control: RESUME - %s", message.c_str());
      break;

    case follow_me::srv::StateControl::Request::STATE_FINISH:
      success = state_machine_->finish();
      message = success ? "Finished successfully" : "Failed to finish";
      if (success) {
        stopRobot();
        template_pool_->reset();
        has_target_ = false;
      }
      RCLCPP_INFO(this->get_logger(), "State control: FINISH - %s", message.c_str());
      break;

    default:
      success = false;
      message = "Invalid state command";
      RCLCPP_WARN(this->get_logger(), "State control: INVALID - %d", request->set_state);
      break;
  }

  response->success = success;
  response->message = message;
}

std::vector<BoundingBox> PersonTrackerNode::extractPersonDetections(
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr & detection_msg)
{
  std::vector<BoundingBox> person_bboxes;
  std::string person_class_id_str = std::to_string(person_class_id_);

  for (const auto & detection : detection_msg->detections) {
    // Check if it's a person class
    bool is_person = false;
    for (const auto & result : detection.results) {
      if (!result.hypothesis.class_id.empty() &&
          (result.hypothesis.class_id == "person" ||
              result.hypothesis.class_id == person_class_id_str)) {
        is_person = true;
        break;
      }
    }

    if (!is_person) {
      continue;
    }

    // Extract bounding box from detection (based on detection_image_width x detection_image_height)
    const auto & bbox_msg = detection.bbox;
    int x1_det = static_cast<int>(bbox_msg.center.position.x - bbox_msg.size_x / 2);
    int y1_det = static_cast<int>(bbox_msg.center.position.y - bbox_msg.size_y / 2);
    int x2_det = static_cast<int>(bbox_msg.center.position.x + bbox_msg.size_x / 2);
    int y2_det = static_cast<int>(bbox_msg.center.position.y + bbox_msg.size_y / 2);

    // Convert detection coordinates from detection image size to original image size
    // Scale factors: original_size / detection_size
    float scale_x = static_cast<float>(latest_rgb_.cols) / detection_image_width_;
    float scale_y = static_cast<float>(latest_rgb_.rows) / detection_image_height_;

    int x1 = static_cast<int>(x1_det * scale_x);
    int y1 = static_cast<int>(y1_det * scale_y);
    int x2 = static_cast<int>(x2_det * scale_x);
    int y2 = static_cast<int>(y2_det * scale_y);

    // Clamp coordinates to image bounds
    x1 = std::max(0, std::min(x1, latest_rgb_.cols - 1));
    y1 = std::max(0, std::min(y1, latest_rgb_.rows - 1));
    x2 = std::max(0, std::min(x2, latest_rgb_.cols - 1));
    y2 = std::max(0, std::min(y2, latest_rgb_.rows - 1));

    float score = detection.results.empty() ?
                      1.0f :
                      static_cast<float>(detection.results[0].hypothesis.score);

    person_bboxes.emplace_back(x1, y1, x2, y2, score);
  }

  return person_bboxes;
}

bool PersonTrackerNode::tryInitializeTarget()
{
  if (current_detections_.empty()) {
    return false;
  }

  // Filter candidate targets
  struct Candidate
  {
    BoundingBox bbox;
    float distance;
  };
  std::vector<Candidate> candidates;

  int image_width = current_rgb_.cols;
  int image_height = current_rgb_.rows;

  // Define center region (50% of image size in each direction)
  int center_x_min = image_width / 4;
  int center_x_max = image_width * 3 / 4;
  int center_y_min = image_height / 4;
  int center_y_max = image_height * 3 / 4;

  for (const auto & bbox : current_detections_) {
    // Calculate depth
    auto depth_opt =
        depth_processor_->computeDepthMedian(current_depth_, bbox, image_width, image_height);
    if (!depth_opt.has_value()) {
      continue;
    }

    float distance = depth_opt.value();
    // Check distance range
    if (distance < min_init_distance_ || distance > max_init_distance_) {
      RCLCPP_DEBUG(this->get_logger(), "Distance is  (%f), not initialize.", distance);
      continue;
    }

    // Check if detection box center is in image center region
    int bbox_center_x = bbox.centerX();
    int bbox_center_y = bbox.centerY();

    if (bbox_center_x >= center_x_min && bbox_center_x <= center_x_max &&
        bbox_center_y >= center_y_min && bbox_center_y <= center_y_max) {
      candidates.push_back({ bbox, distance });
    } else {
      RCLCPP_DEBUG(this->get_logger(), "Target is not in the center region, not initialize.");
    }
  }

  // Initialize only if there is exactly one candidate
  if (candidates.size() != 1) {
    if (candidates.size() > 1) {
      RCLCPP_DEBUG(this->get_logger(),
          "Multiple candidates (%zu) in center region, waiting for single target",
          candidates.size());
    }
    return false;
  }

  // Initialize target
  const auto & candidate = candidates[0];
  target_bbox_ = candidate.bbox;
  has_target_ = true;

  // Extract initial feature
  cv::Rect roi(target_bbox_.x1, target_bbox_.y1, target_bbox_.width(), target_bbox_.height());
  cv::Mat target_image = current_rgb_(roi);

  std::vector<float> initial_feature;
  if (extractReIDFeature(target_image, initial_feature)) {
    template_pool_->initialize(initial_feature);
    RCLCPP_INFO(this->get_logger(), "Target initialized at distance %.2f m", candidate.distance);
    return true;
  }

  return false;
}

bool PersonTrackerNode::performReIDMatching()
{
  if (current_detections_.empty()) {
    stopRobot();
    last_match_info_.clear();
    last_matched_ = false;
    return false;
  }

  // Sort by IOU and compute IOU for each detection
  std::vector<std::pair<float, BoundingBox>> iou_bboxes;
  for (const auto & bbox : current_detections_) {
    float iou = DepthProcessor::computeIOU(bbox, target_bbox_);
    iou_bboxes.push_back({ iou, bbox });
  }
  std::sort(iou_bboxes.begin(), iou_bboxes.end(),
      [](const auto & a, const auto & b) { return a.first > b.first; });

  // Clear previous match info
  last_match_info_.clear();

  auto reid_start_time = std::chrono::steady_clock::now();
  auto max_reid_duration = std::chrono::milliseconds(max_processing_time_ms_);

  bool matched = false;
  size_t reid_count = 0;

  for (const auto & [iou, bbox] : iou_bboxes) {
    auto elapsed = std::chrono::steady_clock::now() - reid_start_time;
    if (elapsed >= max_reid_duration) {
      RCLCPP_WARN(this->get_logger(), "Re-ID timeout after %zu attempts (%.1f ms)", reid_count,
          std::chrono::duration<double, std::milli>(elapsed).count());
      break;
    }

    reid_count++;

    cv::Rect roi(bbox.x1, bbox.y1, bbox.width(), bbox.height());
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > current_rgb_.cols ||
        roi.y + roi.height > current_rgb_.rows) {
      continue;
    }

    cv::Mat candidate_image = current_rgb_(roi);
    std::vector<float> feature;

    if (!extractReIDFeature(candidate_image, feature)) {
      continue;
    }

    float sim_initial = 1.0f;
    computeSimilarity(feature, template_pool_->getInitialTemplate(), sim_initial);

    float sim_latest = 1.0f;
    bool has_latest = template_pool_->hasLatestTemplate();
    if (has_latest) {
      auto latest_template = template_pool_->getLatestTemplate();
      computeSimilarity(feature, latest_template.value(), sim_latest);
    }

    bool is_matched = template_pool_->match(feature, sim_initial, sim_latest);

    // Store match info for visualization
    MatchInfo info;
    info.bbox = bbox;
    info.iou = iou;
    info.sim_initial = sim_initial;
    info.sim_latest = sim_latest;
    info.matched = is_matched;
    info.index = reid_count;
    last_match_info_.push_back(info);

    RCLCPP_DEBUG(this->get_logger(),
        "[REID] Candidate #%zu | IOU=%.3f, Sim_Initial=%.3f, Sim_Latest=%.3f | Match=%s",
        reid_count, iou, sim_initial, sim_latest, is_matched ? "YES" : "NO");

    if (is_matched) {
      target_bbox_ = bbox;
      matched = true;

      template_pool_->tryUpdate(feature, std::min(sim_initial, sim_latest));
      RCLCPP_DEBUG(this->get_logger(), "[REID] ✓ MATCHED after %zu attempts ", reid_count);
      break;
    }
  }

  last_matched_ = matched;

  if (matched) {
    computeAndPublishControl();
    return true;
  } else {
    RCLCPP_DEBUG(
        this->get_logger(), "Tracking lost after %zu Re-ID attempts, stopping robot", reid_count);
    stopRobot();
    last_distance_ = 0.0f;
    last_angle_ = 0.0f;
    return false;
  }
}

std::vector<BoundingBox> PersonTrackerNode::sortByIOU(const std::vector<BoundingBox> & bboxes)
{
  if (!has_target_ || bboxes.empty()) {
    return bboxes;
  }

  // Calculate IOU between each bbox and target
  std::vector<std::pair<float, BoundingBox>> iou_bboxes;
  for (const auto & bbox : bboxes) {
    float iou = DepthProcessor::computeIOU(bbox, target_bbox_);
    iou_bboxes.push_back({ iou, bbox });
  }

  // Sort by IOU in descending order
  std::sort(iou_bboxes.begin(), iou_bboxes.end(),
      [](const auto & a, const auto & b) { return a.first > b.first; });

  // Extract sorted bboxes
  std::vector<BoundingBox> sorted;
  for (const auto & pair : iou_bboxes) {
    sorted.push_back(pair.second);
  }

  return sorted;
}

void PersonTrackerNode::computeAndPublishControl()
{
  if (!has_target_) {
    return;
  }

  // Calculate target distance
  int image_width = current_rgb_.cols;
  int image_height = current_rgb_.rows;
  auto depth_opt =
      depth_processor_->computeDepthMedian(current_depth_, target_bbox_, image_width, image_height);
  if (!depth_opt.has_value()) {
    RCLCPP_WARN(this->get_logger(), "Invalid depth");
    stopRobot();
    return;
  }

  float current_distance = depth_opt.value();

  // Store for visualization
  last_distance_ = current_distance;
  last_angle_ =
      DepthProcessor::computeAngleToTarget(target_bbox_, current_rgb_.cols, fov_horizontal_);

  // Calculate errors
  double distance_error = current_distance - target_distance_;
  double angle_error = last_angle_;

  RCLCPP_DEBUG(this->get_logger(), "[DEPTH] Current=%.2fm, Target=%.2fm", current_distance,
      target_distance_);

  RCLCPP_DEBUG(this->get_logger(), "[ANGLE] Error=%.3f rad (%.1f deg)", angle_error,
      angle_error * 180.0 / M_PI);

  // Calculate time interval
  auto current_time = std::chrono::steady_clock::now();
  double dt = 0.1;  // Default 100ms
  if (processed_frames_ > 1) {
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_process_time_);
    dt = duration.count() / 1000.0;
  }

  // PID control
  double linear_vel, angular_vel;
  pid_controller_->compute(distance_error, angle_error, dt, linear_vel, angular_vel);

  // Print PID control results
  RCLCPP_DEBUG(this->get_logger(),
      "[CONTROL] Linear_vel=%.3f m/s (%s), Angular_vel=%.3f rad/s (%s)", linear_vel,
      (linear_vel > 0) ? "FORWARD" :
      (linear_vel < 0) ? "BACKWARD" :
                         "STOP",
      angular_vel,
      (angular_vel > 0) ? "LEFT" :
      (angular_vel < 0) ? "RIGHT" :
                          "STRAIGHT");

  // Publish control command
  auto cmd = geometry_msgs::msg::Twist();
  cmd.linear.x = linear_vel;
  cmd.angular.z = angular_vel;
  cmd_vel_pub_->publish(cmd);
}

void PersonTrackerNode::stopRobot()
{
  auto cmd = geometry_msgs::msg::Twist();
  cmd.linear.x = 0.0;
  cmd.angular.z = 0.0;
  cmd_vel_pub_->publish(cmd);
}

void PersonTrackerNode::publishVisualization()
{
  if (current_rgb_.empty() || !has_target_) {
    return;
  }

  cv::Mat vis_image = current_rgb_.clone();

  // Draw only tracking target and boxes with IOU > 0
  for (const auto & info : last_match_info_) {
    cv::Scalar color = info.matched ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    cv::Scalar bg_color = cv::Scalar(0, 0, 0);  // Black background for text
    int thickness = info.matched ? 3 : 2;

    cv::rectangle(vis_image, cv::Point(info.bbox.x1, info.bbox.y1),
        cv::Point(info.bbox.x2, info.bbox.y2), color, thickness);

    // Draw matching scores above the bounding box with backgrounds
    int y_offset = info.bbox.y1 - 10;  // Start above the box

    // Draw index number with background
    std::string idx_text = "#" + std::to_string(info.index);
    cv::Size text_size = cv::getTextSize(idx_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, nullptr);
    cv::Point idx_pos(info.bbox.x1, y_offset);
    cv::rectangle(vis_image, cv::Point(idx_pos.x - 2, idx_pos.y - text_size.height - 2),
        cv::Point(idx_pos.x + text_size.width + 2, idx_pos.y + 2), bg_color, -1);
    cv::putText(vis_image, idx_text, idx_pos, cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);

    y_offset -= 30;  // Move further up
    std::string sim_init_text = "Init:" + std::to_string(info.sim_initial).substr(0, 4);
    text_size = cv::getTextSize(sim_init_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, nullptr);
    cv::Point init_pos(info.bbox.x1, y_offset);
    cv::rectangle(vis_image, cv::Point(init_pos.x - 2, init_pos.y - text_size.height - 2),
        cv::Point(init_pos.x + text_size.width + 2, init_pos.y + 2), bg_color, -1);
    cv::putText(vis_image, sim_init_text, init_pos, cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);

    y_offset -= 30;  // Move further up
    std::string sim_latest_text = "Latest:" + std::to_string(info.sim_latest).substr(0, 4);
    text_size = cv::getTextSize(sim_latest_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, nullptr);
    cv::Point latest_pos(info.bbox.x1, y_offset);
    cv::rectangle(vis_image, cv::Point(latest_pos.x - 2, latest_pos.y - text_size.height - 2),
        cv::Point(latest_pos.x + text_size.width + 2, latest_pos.y + 2), bg_color, -1);
    cv::putText(vis_image, sim_latest_text, latest_pos, cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);

    y_offset -= 30;  // Move further up
    std::string match_text = info.matched ? "Match:YES" : "Match:NO";
    text_size = cv::getTextSize(match_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, nullptr);
    cv::Point match_pos(info.bbox.x1, y_offset);
    cv::rectangle(vis_image, cv::Point(match_pos.x - 2, match_pos.y - text_size.height - 2),
        cv::Point(match_pos.x + text_size.width + 2, match_pos.y + 2), bg_color, -1);
    cv::putText(vis_image, match_text, match_pos, cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
  }

  // Draw unified threshold information at top with background
  cv::Scalar bg_color = cv::Scalar(0, 0, 0);  // Black background for text
  int text_y = 25;
  std::string thresh_text = "Thresholds - Match:" + std::to_string(match_threshold_).substr(0, 4) +
                            " Update:" + std::to_string(template_update_threshold_).substr(0, 4) +
                            " Drift:" + std::to_string(drift_threshold_).substr(0, 4);
  cv::Size text_size = cv::getTextSize(thresh_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, nullptr);
  cv::Point thresh_pos(10, text_y);
  cv::rectangle(vis_image, cv::Point(thresh_pos.x - 2, thresh_pos.y - text_size.height - 2),
      cv::Point(thresh_pos.x + text_size.width + 2, thresh_pos.y + 2), bg_color, -1);
  cv::putText(
      vis_image, thresh_text, thresh_pos, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

  // If matched, draw distance and angle with backgrounds
  if (last_matched_) {
    text_y += 35;
    std::string dist_text = "Distance: " + std::to_string(last_distance_).substr(0, 4) + "m";
    text_size = cv::getTextSize(dist_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, nullptr);
    cv::Point dist_pos(10, text_y);
    cv::rectangle(vis_image, cv::Point(dist_pos.x - 2, dist_pos.y - text_size.height - 2),
        cv::Point(dist_pos.x + text_size.width + 2, dist_pos.y + 2), bg_color, -1);
    cv::putText(
        vis_image, dist_text, dist_pos, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    text_y += 35;
    float angle_deg = last_angle_ * 180.0f / M_PI;
    std::string angle_text = "Angle: " + std::to_string(angle_deg).substr(0, 5) + " deg";
    text_size = cv::getTextSize(angle_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, nullptr);
    cv::Point angle_pos(10, text_y);
    cv::rectangle(vis_image, cv::Point(angle_pos.x - 2, angle_pos.y - text_size.height - 2),
        cv::Point(angle_pos.x + text_size.width + 2, angle_pos.y + 2), bg_color, -1);
    cv::putText(
        vis_image, angle_text, angle_pos, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
  }

  try {
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", vis_image).toImageMsg();
    vis_pub_->publish(*msg);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

void PersonTrackerNode::processingLoop()
{
  RCLCPP_INFO(this->get_logger(), "Processing thread started");

  while (process_thread_running_) {
    // Wait for new data
    std::unique_lock<std::mutex> lock(data_mutex_);
    data_cv_.wait(lock, [this] { return has_new_data_ || !process_thread_running_; });

    if (!process_thread_running_) {
      RCLCPP_INFO(this->get_logger(), "Processing thread shutting down");
      break;
    }

    if (!has_new_data_) {
      RCLCPP_DEBUG(this->get_logger(), "Woke up but no new data, continuing");
      continue;
    }

    // Copy latest data
    cv::Mat rgb = latest_rgb_.clone();
    cv::Mat depth = latest_depth_.clone();
    auto detections = latest_detections_;
    has_new_data_ = false;
    lock.unlock();

    if (rgb.empty() || depth.empty()) {
      RCLCPP_WARN(this->get_logger(), "Received empty RGB or depth image");
      continue;
    }

    // Process data
    auto start_time = std::chrono::steady_clock::now();

    try {
      processData(rgb, depth, detections);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Exception in processData: %s", e.what());
    }

    auto end_time = std::chrono::steady_clock::now();
    auto processing_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (debug_mode_ && processing_duration.count() > max_processing_time_ms_) {
      RCLCPP_WARN(this->get_logger(), "Processing took %ld ms (limit: %d ms)",
          processing_duration.count(), max_processing_time_ms_);
    }

    processed_frames_++;
    last_process_time_ = end_time;
  }

  RCLCPP_INFO(this->get_logger(), "Processing thread stopped");
}

void PersonTrackerNode::processData(const cv::Mat & rgb,
    const cv::Mat & depth,
    const std::vector<BoundingBox> & detections)
{
  // Update current data
  current_rgb_ = rgb;
  current_depth_ = depth;
  current_detections_ = detections;

  // Process based on whether target is initialized
  if (!has_target_) {
    // Not initialized, try to initialize target
    tryInitializeTarget();
  } else {
    // Already initialized, perform tracking
    performReIDMatching();
  }

  // Publish visualization if debug mode is enabled
  if (debug_mode_) {
    publishVisualization();
  }
}

bool PersonTrackerNode::extractReIDFeature(const cv::Mat & image, std::vector<float> & feature)
{
  feature.clear();

  if (image.empty()) {
    RCLCPP_WARN(this->get_logger(), "Empty image passed to extractReIDFeature");
    return false;
  }

  if (!extract_feature_client_) {
    RCLCPP_ERROR(this->get_logger(), "ExtractFeature client is not initialized");
    return false;
  }

  if (!extract_feature_client_->wait_for_service(std::chrono::seconds(2))) {
    RCLCPP_ERROR(this->get_logger(), "ExtractFeature service unavailable");
    return false;
  }

  auto cv_img = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image);
  auto request = std::make_shared<qrb_ros_people_reid::srv::ExtractFeature::Request>();
  request->image = *cv_img.toImageMsg();

  auto future = extract_feature_client_->async_send_request(request);

  // Wait for the future to complete without using spin_until_future_complete
  // to avoid adding the node to multiple executors
  auto status = future.wait_for(std::chrono::seconds(5));
  if (status != std::future_status::ready) {
    RCLCPP_ERROR(this->get_logger(), "Failed to call ExtractFeature service (timeout)");
    return false;
  }

  const auto response = future.get();
  if (!response->success) {
    RCLCPP_WARN(this->get_logger(), "ExtractFeature failed: %s", response->message.c_str());
    return false;
  }

  feature = response->feature;
  return true;
}

bool PersonTrackerNode::computeSimilarity(const std::vector<float> & feature1,
    const std::vector<float> & feature2,
    float & similarity)
{
  similarity = 0.0f;

  if (feature1.empty() || feature2.empty()) {
    RCLCPP_WARN(this->get_logger(), "Empty feature vector passed to computeSimilarity");
    return false;
  }

  if (!compute_similarity_client_) {
    RCLCPP_ERROR(this->get_logger(), "ComputeSimilarity client is not initialized");
    return false;
  }

  if (!compute_similarity_client_->wait_for_service(std::chrono::seconds(2))) {
    RCLCPP_ERROR(this->get_logger(), "ComputeSimilarity service unavailable");
    return false;
  }

  auto request = std::make_shared<qrb_ros_people_reid::srv::ComputeSimilarity::Request>();
  request->feature1 = feature1;
  request->feature2 = feature2;

  auto future = compute_similarity_client_->async_send_request(request);

  // Wait for the future to complete without using spin_until_future_complete
  // to avoid adding the node to multiple executors
  auto status = future.wait_for(std::chrono::seconds(5));
  if (status != std::future_status::ready) {
    RCLCPP_ERROR(this->get_logger(), "Failed to call ComputeSimilarity service (timeout)");
    return false;
  }

  const auto response = future.get();
  if (!response->success) {
    RCLCPP_WARN(this->get_logger(), "ComputeSimilarity failed: %s", response->message.c_str());
    return false;
  }

  similarity = response->similarity;
  return true;
}

}  // namespace follow_me

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<follow_me::PersonTrackerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
