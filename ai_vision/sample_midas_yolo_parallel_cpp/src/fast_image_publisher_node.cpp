// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// FastImagePublisherNode: loads an image once at startup and publishes it
// at the requested rate without per-frame file I/O or JPEG decode overhead.

#include <chrono>
#include <memory>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace sample_midas_yolo_parallel_cpp
{

class FastImagePublisherNode : public rclcpp::Node
{
public:
  explicit FastImagePublisherNode(const rclcpp::NodeOptions & options)
  : Node("fast_image_publisher_node", options)
  {
    const std::string filename = declare_parameter("filename", "");
    const double rate_hz       = declare_parameter("rate",     30.0);

    if (filename.empty()) {
      RCLCPP_ERROR(get_logger(), "filename parameter is required");
      return;
    }

    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
    if (img.empty()) {
      RCLCPP_ERROR(get_logger(), "Failed to load image: %s", filename.c_str());
      return;
    }

    // Pre-build the Image message once — reuse on every publish
    msg_.header.frame_id = "camera";
    msg_.height   = static_cast<uint32_t>(img.rows);
    msg_.width    = static_cast<uint32_t>(img.cols);
    msg_.encoding = "bgr8";
    msg_.step     = static_cast<uint32_t>(img.step);
    msg_.is_bigendian = false;
    const size_t nbytes = img.total() * img.elemSize();
    msg_.data.resize(nbytes);
    std::memcpy(msg_.data.data(), img.data, nbytes);

    pub_ = create_publisher<sensor_msgs::msg::Image>("image_raw", 10);

    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / rate_hz));
    timer_ = create_wall_timer(period, [this]() { publish(); });

    RCLCPP_INFO(get_logger(),
        "FastImagePublisher: %s (%ux%u) at %.1f Hz",
        filename.c_str(), msg_.width, msg_.height, rate_hz);
  }

private:
  void publish()
  {
    msg_.header.stamp = this->now();
    pub_->publish(msg_);
  }

  sensor_msgs::msg::Image msg_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace sample_midas_yolo_parallel_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(
    sample_midas_yolo_parallel_cpp::FastImagePublisherNode)
