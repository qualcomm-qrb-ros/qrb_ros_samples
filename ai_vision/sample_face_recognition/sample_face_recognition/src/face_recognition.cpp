// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// Copyright (c) 2026, Qualcomm Innovation Center, Inc.
// All rights reserved.

#include "sample_face_recognition/face_recognition.hpp"

#include <algorithm>

using namespace cv;
using namespace std;
using namespace std::chrono;
using namespace std::placeholders;

namespace sample_face_recognition
{
RecognitionNode::RecognitionNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("RecognitionNode", options)
{
  RCLCPP_INFO(this->get_logger(), " == start face recognition process == ");

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>("input_faces", 10,
      std::bind(&RecognitionNode::recognition_callback, this, std::placeholders::_1), sub_options);

  target_sub_ = this->create_subscription<sensor_msgs::msg::Image>("target_face", 10,
      std::bind(&RecognitionNode::target_callback, this, std::placeholders::_1), sub_options);

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  faces_pub_ = this->create_publisher<sample_face_customed_msgs::msg::FaceArrayMessage>(
      "detected_faces", 10, pub_options);

  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  sub_options.callback_group = callback_group_;

  fd_model_ = this->declare_parameter<std::string>("fd_model",
      "/usr/share/sample_face_recognition_recognition/model/face_detection_yunet_2023dec.onnx");
  fr_model_ = this->declare_parameter<std::string>("fr_model",
      "/usr/share/sample_face_recognition_recognition/model/face_recognition_sface_2021dec.onnx");
  database_path_ = this->declare_parameter<std::string>(
      "database_path", "/usr/share/sample_face_recognition_recognition/data");
  image_data_path_ = this->declare_parameter<std::string>("image_data_path", "");
  score_threshold_ = this->declare_parameter<float>("score_threshold", 0.9);
  nms_threshold_ = this->declare_parameter<float>("nms_threshold", 0.3);
  scale_ = this->declare_parameter<float>("scale", 1.0);
  top_k_ = this->declare_parameter<int>("top_k", 5000);
  overlay_ = this->declare_parameter<int>("overlay", 1);
  fps_ = this->declare_parameter<int>("fps_max", 5);
  cosine_similar_thresh_ = this->declare_parameter<float>("similar_threshold", 0.363);

  timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / fps_), std::bind(&RecognitionNode::timer_callback, this));

  if (overlay_) {
    image_pub_ =
        this->create_publisher<sensor_msgs::msg::Image>("detected_faces_overlay", 10, pub_options);
  }

  RCLCPP_INFO(this->get_logger(), "Loading model %s", fd_model_.c_str());
  RCLCPP_INFO(this->get_logger(), "Loading model %s", fr_model_.c_str());
  face_detect_all_ = std::make_shared<sample_face_recognition::FaceDetect>(
      fd_model_, Size(320, 320), score_threshold_, nms_threshold_, top_k_, fr_model_);

  if (!image_data_path_.empty()) {
    face_detect_all_->update_database(image_data_path_);
    RCLCPP_INFO(this->get_logger(), "taget image path=%s", image_data_path_.c_str());
  }

  database_path_ = image_data_path_.empty() ? database_path_ : image_data_path_;
  RCLCPP_INFO(this->get_logger(), "database path=%s", database_path_.c_str());
  face_detect_all_->load_database(database_path_);

  RCLCPP_INFO(this->get_logger(), "similar_threshold = %0.3f", cosine_similar_thresh_);
  face_detect_all_->set_similar_thresh(cosine_similar_thresh_);

  service_ = this->create_service<sample_face_customed_msgs::srv::QueryName>(
      "/sample_face_query_name", std::bind(&RecognitionNode::service_callback, this, _1, _2, _3),
      rclcpp::ServicesQoS(), callback_group_);

  RCLCPP_INFO(this->get_logger(), "start QueryName service");

  count_ = 0;
}

void RecognitionNode::timer_callback()
{
  status_ = true;
}

void RecognitionNode::service_callback(const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<sample_face_customed_msgs::srv::QueryName::Request> request,
    std::shared_ptr<sample_face_customed_msgs::srv::QueryName::Response> response)
{
  (void)request_header;
  response->result = face_detect_all_->check_database(request->name);
  RCLCPP_INFO(
      rclcpp::get_logger("rclcpp"), "Incoming request query target: %s", request->name.c_str());
  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "sending back response: [%d]", response->result);
}

inline bool compare(sample_face_customed_msgs::msg::FaceMessage & a,
    sample_face_customed_msgs::msg::FaceMessage & b)
{
  return (a.width_bbox * a.height_bbox) > (b.width_bbox * b.height_bbox);
}

void RecognitionNode::publish_faces_topic(cv::Mat & faces_src,
    std::vector<std::string> & face_names)
{
  auto message = sample_face_customed_msgs::msg::FaceArrayMessage();
  message.header.stamp = this->get_clock()->now();
  message.header.frame_id = std::to_string(count_);

  for (int i = 0; i < faces_src.rows; ++i) {
    sample_face_customed_msgs::msg::FaceMessage element;
    element.name = face_names[i];
    element.x_bbox = int(faces_src.at<float>(i, 0));
    element.y_bbox = int(faces_src.at<float>(i, 1));
    element.width_bbox = int(faces_src.at<float>(i, 2));
    element.height_bbox = int(faces_src.at<float>(i, 3));
    element.x_right_eye = int(faces_src.at<float>(i, 4));
    element.y_right_eye = int(faces_src.at<float>(i, 5));
    element.x_left_eye = int(faces_src.at<float>(i, 6));
    element.y_left_eye = int(faces_src.at<float>(i, 7));
    element.x_nose_tip = int(faces_src.at<float>(i, 8));
    element.y_nose_tip = int(faces_src.at<float>(i, 9));
    element.x_right_corner = int(faces_src.at<float>(i, 10));
    element.y_right_corner = int(faces_src.at<float>(i, 11));
    element.x_left_corner = int(faces_src.at<float>(i, 12));
    element.y_left_corner = int(faces_src.at<float>(i, 13));
    element.face_score = faces_src.at<float>(i, 14);
    RCLCPP_INFO(this->get_logger(), "detecting %d %-16s score=%0.3f size=%d", i + 1,
        face_names[i].size() ? face_names[i].c_str() : "person", element.face_score,
        element.width_bbox * element.height_bbox);
    message.data.push_back(element);
  }

  std::sort(message.data.begin(), message.data.end(), compare);
  faces_pub_->publish(message);
}

void RecognitionNode::recognition_callback(sensor_msgs::msg::Image::SharedPtr msg)
{
  RCLCPP_DEBUG(this->get_logger(), " == Received image run face recognition == ");
  count_++;

  if (status_) {
    status_ = false;
    cv_ptr_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    image_src_ = cv_ptr_->image;
    std::vector<std::string> face_names;
    cv::Mat faces_src;
    face_detect_all_->detect(image_src_, faces_src, face_names, overlay_);

    publish_faces_topic(faces_src, face_names);
    if (overlay_)
      image_pub_->publish(*(cv_ptr_->toImageMsg()));
  }
}

bool create_directory(const std::string & path)
{
  struct stat info;
  if (stat(path.c_str(), &info) != 0) {
    return mkdir(path.c_str(), 0777) == 0;
  } else if (info.st_mode & S_IFDIR) {
    return true;
  } else {
    return false;
  }
}

void RecognitionNode::target_callback(sensor_msgs::msg::Image::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), " == Received a target face == ");
  try {
    cv::Mat cv_image = cv_bridge::toCvShare(msg, "bgr8")->image;
    std::string frame_id = msg->header.frame_id;
    std::string file_name = image_data_path_ + "/" + frame_id + ".jpg";
    std::string test_name = image_data_path_ + "/test/" + frame_id + ".jpg";
    std::string test_dir = image_data_path_ + "/test";

    create_directory(test_dir);
    cv::imwrite(test_name, cv_image);
    if (face_detect_all_->add_database(test_name)) {
      std::rename(test_name.c_str(), file_name.c_str());
      RCLCPP_INFO(this->get_logger(), "Saved image: %s", file_name.c_str());
    } else {
      std::remove(test_name.c_str());
    }

  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  } catch (const cv::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "OpenCV exception: %s", e.what());
  }
}

}  // namespace sample_face_recognition

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(sample_face_recognition::RecognitionNode)