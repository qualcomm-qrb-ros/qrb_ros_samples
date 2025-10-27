/* Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#include <controller.h>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tracker.h>

std::string model_param_file =
    std::string("/usr/share/follow-me/model/FastestDet.param");
std::string model_bin_file =
    std::string("/usr/share/follow-me/model/FastestDet.bin");
std::string label_file = std::string("/usr/share/follow-me/model/labels.txt");

// std::string model_param_file = std::string(
// "/opt/qcom/qirf-sdk/data/model/FastestDet.param" ); std::string
// model_bin_file = std::string( "/opt/qcom/qirf-sdk/data/model/FastestDet.bin"
// ); std::string label_file = std::string(
// "/opt/qcom/qirf-sdk/data/model/labels.txt");

class FollowMeNode : public rclcpp::Node {
public:
  FollowMeNode()
      : Node("follow_me_node"), controller_(),
        tracker_(TRACKING_TYPE_CV, TRACKING_CV_PERSON, model_param_file,
                 model_bin_file, label_file) {
    rclcpp::QoS qos(10);
    qos.best_effort();
    controller_.speed_pub =
        this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 3);
    color_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/color/image_raw", qos,
        std::bind(&FollowMeNode::color_callback, this, std::placeholders::_1));
    depth_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/depth/image_raw", qos,
        std::bind(&FollowMeNode::depth_callback, this, std::placeholders::_1));
    intrinsics_subscription_ =
        this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/color/camera_info", 10,
            std::bind(&FollowMeNode::intrinsic_callback, this,
                      std::placeholders::_1));
    timer_ =
        this->create_wall_timer(std::chrono::milliseconds(200),
                                std::bind(&FollowMeNode::timer_callback, this));
  }

private:
  void color_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::cout << "hello, color img received!" << std::endl;
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
      RCLCPP_INFO(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    tracker_.getImageOnce(cv_ptr->image);
  }

  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        cv_ptr =
            cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
      } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        cv_ptr =
            cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
      } else {
        RCLCPP_WARN(this->get_logger(), "Unsupported image encoding: %s",
                    msg->encoding.c_str());
        return;
      }
    } catch (cv_bridge::Exception &e) {
      RCLCPP_INFO(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    tracker_.getDepthOnce(cv_ptr->image);
  }

  void intrinsic_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    tracker_.setIntrinsics(msg);
  }

  void timer_callback() {
    float theta = 0.0;
    float d = 0.0;
    if (tracker_.trackingOnce()) {
      std::cout << "trackingOnce return true!" << std::endl;
      if (tracker_.dist__m > 3.0)
        controller_.controlOnce(tracker_.theta__m, 1.5);
      controller_.controlOnce(tracker_.theta__m, tracker_.dist__m);
      // controller_.controlOnce(theta,d);
    } else {
      controller_.stop();
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      intrinsics_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
  bool track_result_;
  MyTracker tracker_;
  Controller controller_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<FollowMeNode>());

  rclcpp::shutdown();

  return 0;
}
