// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#ifndef QRB_ROS_TTS__SPEAKER_NODE_HPP_
#define QRB_ROS_TTS__SPEAKER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "audio_types.hpp"
#include "qrb_ros_tts/visibility.hpp"
#include "synthesizer.hpp"

class SpeakerNode : public rclcpp::Node
{
public:
  QRB_ROS_TTS_PUBLIC
  explicit SpeakerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  ~SpeakerNode();

private:
  void handle_speech_request(const std_msgs::msg::String::SharedPtr msg);

  audio::Synthesizer synthesizer_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
};

#endif  // QRB_ROS_TTS__SPEAKER_NODE_HPP_
