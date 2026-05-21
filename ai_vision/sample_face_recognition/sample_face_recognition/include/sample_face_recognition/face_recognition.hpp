// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// Copyright (c) 2026, Qualcomm Innovation Center, Inc.
// All rights reserved.

#ifndef SAMPLE_FACE_RECOGNITION_HPP_
#define SAMPLE_FACE_RECOGNITION_HPP_

#include <sys/stat.h>

#include <chrono>
#include <cv_bridge/cv_bridge.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <vector>

#include "sample_face_customed_msgs/msg/face_array_message.hpp"
#include "sample_face_customed_msgs/srv/query_name.hpp"
#include "sample_face_recognition/face_detect.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

namespace sample_face_recognition
{
class RecognitionNode : public rclcpp::Node
{
public:
  explicit RecognitionNode(const rclcpp::NodeOptions &);

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_{ nullptr };
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr target_sub_{ nullptr };
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_{ nullptr };
  rclcpp::Publisher<sample_face_customed_msgs::msg::FaceArrayMessage>::SharedPtr faces_pub_{
    nullptr
  };
  rclcpp::Service<sample_face_customed_msgs::srv::QueryName>::SharedPtr service_{ nullptr };
  rclcpp::CallbackGroup::SharedPtr callback_group_{ nullptr };
  rclcpp::TimerBase::SharedPtr timer_{ nullptr };

  std::string fd_model_;
  std::string fr_model_;
  std::string database_path_;
  std::string image_data_path_;
  float score_threshold_;
  float nms_threshold_;
  float scale_;
  int top_k_;
  bool overlay_;
  bool status_;
  int run_per_count_;
  int fps_;
  size_t count_{ 0 };
  float cosine_similar_thresh_{ 0.363 };

  cv_bridge::CvImagePtr cv_ptr_;
  cv::Mat image_src_;

  std::shared_ptr<sample_face_recognition::FaceDetect> face_detect_all_{ nullptr };

  void timer_callback();
  void recognition_callback(sensor_msgs::msg::Image::SharedPtr msg);
  void target_callback(sensor_msgs::msg::Image::SharedPtr msg);
  void publish_faces_topic(cv::Mat & faces_src, std::vector<std::string> & face_names);
  void service_callback(const std::shared_ptr<rmw_request_id_t> request_header,
      const std::shared_ptr<sample_face_customed_msgs::srv::QueryName::Request> request,
      std::shared_ptr<sample_face_customed_msgs::srv::QueryName::Response> response);
};
}  // namespace sample_face_recognition

#endif  // SAMPLE_FACE_RECOGNITION_HPP_
