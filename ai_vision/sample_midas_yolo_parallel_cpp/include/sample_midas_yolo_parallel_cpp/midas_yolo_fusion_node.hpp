// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <qrb_ros_tensor_list_msgs/msg/tensor_list.hpp>

namespace sample_midas_yolo_parallel_cpp
{

namespace custom_msg = qrb_ros_tensor_list_msgs::msg;

// ── per-frame state ──────────────────────────────────────────────────────────

struct Detection
{
  int   cls;
  float score;
  float x1, y1, x2, y2;
  std::vector<float> coeff;  // 32 mask coefficients
};

struct PendingFrame
{
  int32_t  header_sec{0};
  uint32_t header_nsec{0};
  cv::Mat  image_bgr;
  int      yolo_input_w{640};
  int      yolo_input_h{640};
  double   created_at{0.0};

  // Inference outputs — set by the respective output callbacks
  std::shared_ptr<const custom_msg::TensorList> midas_tensors;
  std::shared_ptr<const custom_msg::TensorList> yolo_tensors;
};

using FrameKey = std::pair<int32_t, uint32_t>;

struct FrameKeyHash
{
  std::size_t operator()(const FrameKey & k) const noexcept
  {
    return std::hash<int64_t>{}(
        (static_cast<int64_t>(k.first) << 32) | static_cast<int64_t>(k.second));
  }
};

// ── node ─────────────────────────────────────────────────────────────────────

class MidasYoloFusionNode : public rclcpp::Node
{
public:
  explicit MidasYoloFusionNode(const rclcpp::NodeOptions & options);
  ~MidasYoloFusionNode() = default;

private:
  // ── parameters ──
  std::string input_topic_;
  std::string midas_input_name_;
  std::string yolo_input_name_;
  int         midas_data_type_{0};
  int         yolo_data_type_{0};
  bool        yolo_pack_uint16_{false};
  int         midas_h_{256}, midas_w_{256};
  int         yolo_h_{640},  yolo_w_{640};
  float       score_thresh_{0.25f};
  float       iou_thresh_{0.45f};
  float       overlay_alpha_{0.45f};
  int         max_pending_{4};

  // ── subscriptions / publishers ──
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr        image_sub_;
  rclcpp::Subscription<custom_msg::TensorList>::SharedPtr         midas_out_sub_;
  rclcpp::Subscription<custom_msg::TensorList>::SharedPtr         yolo_out_sub_;
  rclcpp::Publisher<custom_msg::TensorList>::SharedPtr            midas_in_pub_;
  rclcpp::Publisher<custom_msg::TensorList>::SharedPtr            yolo_in_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           overlay_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           depth_color_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           depth_gray_pub_;

  // ── pending frame map ──
  std::mutex pending_mutex_;
  std::unordered_map<FrameKey, PendingFrame, FrameKeyHash> pending_;
  std::atomic<int64_t> synthetic_seq_{0};

  // ── stats ──
  std::atomic<uint64_t> processed_count_{0};
  rclcpp::Time          last_log_time_;

  // ── callbacks ──
  void image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg);
  void midas_output_callback(custom_msg::TensorList::ConstSharedPtr msg);
  void yolo_output_callback(custom_msg::TensorList::ConstSharedPtr msg);

  // ── helpers ──
  FrameKey extract_key(const std_msgs::msg::Header & hdr);
  FrameKey match_pending_key(const std_msgs::msg::Header & hdr);

  cv::Mat decode_image(sensor_msgs::msg::Image::ConstSharedPtr msg);
  cv::Mat prep_midas(const cv::Mat & bgr);
  cv::Mat prep_yolo(const cv::Mat & bgr);

  custom_msg::TensorList make_tensor_msg(
      const std::string & name, const cv::Mat & data, int data_type,
      const std_msgs::msg::Header & hdr);

  // ── post-processing ──
  void fuse_and_publish(FrameKey key);

  // depth
  void decode_midas_depth(
      const custom_msg::Tensor & tensor, int out_w, int out_h,
      cv::Mat & depth_f32, cv::Mat & depth_gray, cv::Mat & depth_color);

  // yolo
  cv::Mat tensor_to_mat(const custom_msg::Tensor & tensor);
  cv::Mat proto_to_hwc(const cv::Mat & proto);
  std::vector<Detection> decode_yolo_split(
      const cv::Mat & boxes, const cv::Mat & scores,
      const cv::Mat & class_idx, const cv::Mat & coeffs,
      int input_w, int input_h);
  std::vector<Detection> nms(std::vector<Detection> dets);
  void parse_yolo_outputs(
      const custom_msg::TensorList & tensors, int input_w, int input_h,
      std::vector<Detection> & dets_out, cv::Mat & proto_out);

  // overlay
  float depth_p85(const cv::Mat & depth, const cv::Mat & mask);
  void draw_overlay(
      cv::Mat & out, const cv::Mat & depth_f32,
      const std::vector<Detection> & dets, const cv::Mat & proto_hwc,
      int input_w, int input_h);

  // image msg builder
  sensor_msgs::msg::Image mat_to_image_msg(
      const cv::Mat & mat, const std::string & encoding,
      const std_msgs::msg::Header & hdr);
};

}  // namespace sample_midas_yolo_parallel_cpp
