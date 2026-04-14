// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef FOLLOW_ME__PERSON_TRACKER_NODE_HPP_
#define FOLLOW_ME__PERSON_TRACKER_NODE_HPP_

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <memory>
#include <mutex>
#include <qrb_ros_people_reid/srv/compute_similarity.hpp>
#include <qrb_ros_people_reid/srv/extract_feature.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
#include <vector>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "follow_me/depth_processor.hpp"
#include "follow_me/pid_controller.hpp"
#include "follow_me/srv/state_control.hpp"
#include "follow_me/state_machine.hpp"
#include "follow_me/template_pool.hpp"

namespace follow_me
{

/**
 * @brief Person Tracker main node
 */
class PersonTrackerNode : public rclcpp::Node
{
public:
  PersonTrackerNode();
  ~PersonTrackerNode();

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
      sensor_msgs::msg::Image,
      vision_msgs::msg::Detection2DArray>;

  // Subscribers
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub_;
  std::shared_ptr<message_filters::Subscriber<vision_msgs::msg::Detection2DArray>> detection_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  // ROS service clients
  rclcpp::Client<qrb_ros_people_reid::srv::ExtractFeature>::SharedPtr extract_feature_client_;
  rclcpp::Client<qrb_ros_people_reid::srv::ComputeSimilarity>::SharedPtr compute_similarity_client_;

  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr vis_pub_;

  // Services
  rclcpp::Service<follow_me::srv::StateControl>::SharedPtr state_control_service_;

  // Core modules
  std::unique_ptr<TemplatePool> template_pool_;
  std::unique_ptr<DualPIDController> pid_controller_;
  std::unique_ptr<DepthProcessor> depth_processor_;
  std::unique_ptr<StateMachine> state_machine_;

  // Current data
  cv::Mat current_rgb_;
  cv::Mat current_depth_;
  std::vector<BoundingBox> current_detections_;
  BoundingBox target_bbox_;
  bool has_target_;

  // Parameters
  int detection_image_width_;        // Detection model input image width (640)
  int detection_image_height_;       // Detection model input image height (640)
  double target_distance_;           // Target distance (2.0m)
  double min_init_distance_;         // Minimum initialization distance (1.5m)
  double max_init_distance_;         // Maximum initialization distance (3.0m)
  int person_class_id_;              // Person class ID (0 for COCO dataset)
  float template_update_threshold_;  // Template update threshold (0.1)
  float match_threshold_;            // Matching threshold (0.25)
  float drift_threshold_;            // Anti-drift threshold (0.1)
  int max_processing_time_ms_;       // Maximum processing time (200ms)
  bool debug_mode_;                  // Debug mode (publish visualization)

  // PID parameters
  double linear_kp_, linear_ki_, linear_kd_;
  double angular_kp_, angular_ki_, angular_kd_;
  double max_linear_speed_, max_angular_speed_;

  // Camera info
  float fov_horizontal_{ M_PI / 2.0f };  // Default 90 degrees
  bool camera_info_received_{ false };

  // Visualization data
  struct MatchInfo
  {
    BoundingBox bbox;
    float iou;
    float sim_initial;
    float sim_latest;
    bool matched;
    size_t index;
  };
  std::vector<MatchInfo> last_match_info_;
  float last_distance_{ 0.0f };
  float last_angle_{ 0.0f };
  bool last_matched_{ false };

  // Statistics
  std::atomic<uint64_t> processed_frames_{ 0 };
  std::chrono::steady_clock::time_point last_process_time_;

  // Processing thread related
  std::thread processing_thread_;
  std::atomic<bool> process_thread_running_{ true };
  std::mutex data_mutex_;
  std::condition_variable data_cv_;
  bool has_new_data_{ false };

  // Latest data (written by callback thread, read by processing thread)
  cv::Mat latest_rgb_;
  cv::Mat latest_depth_;
  std::vector<BoundingBox> latest_detections_;

  /**
   * @brief Camera info callback
   */
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  /**
   * @brief Synchronized callback
   */
  void syncCallback(const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
      const vision_msgs::msg::Detection2DArray::ConstSharedPtr & detection_msg);

  /**
   * @brief State control service callback
   */
  void stateControlCallback(const std::shared_ptr<follow_me::srv::StateControl::Request> request,
      std::shared_ptr<follow_me::srv::StateControl::Response> response);

  /**
   * @brief Processing thread main loop
   */
  void processingLoop();

  /**
   * @brief Process data (called in processing thread)
   */
  void processData(const cv::Mat & rgb,
      const cv::Mat & depth,
      const std::vector<BoundingBox> & detections);

  /**
   * @brief Try to initialize target
   */
  bool tryInitializeTarget();

  /**
   * @brief Perform Re-ID matching
   */
  bool performReIDMatching();

  /**
   * @brief Sort detection boxes by IOU
   */
  std::vector<BoundingBox> sortByIOU(const std::vector<BoundingBox> & bboxes);

  /**
   * @brief Compute and publish control command
   */
  void computeAndPublishControl();

  /**
   * @brief Stop robot
   */
  void stopRobot();

  /**
   * @brief Publish visualization image
   */
  void publishVisualization();

  /**
   * @brief Extract person detection boxes
   */
  std::vector<BoundingBox> extractPersonDetections(
      const vision_msgs::msg::Detection2DArray::ConstSharedPtr & detection_msg);

  /**
   * @brief Declare parameters
   */
  void declareParameters();

  /**
   * @brief Load parameters
   */
  void loadParameters();

  /**
   * @brief Extract ReID feature from image
   */
  bool extractReIDFeature(const cv::Mat & image, std::vector<float> & feature);

  /**
   * @brief Compute similarity between two feature vectors
   */
  bool computeSimilarity(const std::vector<float> & feature1,
      const std::vector<float> & feature2,
      float & similarity);
};

}  // namespace follow_me

#endif  // FOLLOW_ME__PERSON_TRACKER_NODE_HPP_
