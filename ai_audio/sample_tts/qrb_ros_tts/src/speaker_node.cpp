// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "qrb_ros_tts/speaker_node.hpp"

#include <functional>

#include "rclcpp_components/register_node_macro.hpp"

using audio::Status;

SpeakerNode::SpeakerNode(const rclcpp::NodeOptions & options) : Node("speaker_node", options)
{
  declare_parameter<std::string>("config_file", "");
  declare_parameter<std::string>("output_wav", "output.wav");

  Status st = synthesizer_.initialize();
  if (st != Status::Ok) {
    RCLCPP_ERROR(get_logger(), "Synthesizer init failed (code=%d)", static_cast<int>(st));
    throw std::runtime_error("SpeakerNode: synthesizer initialization failed");
  }

  sub_ = create_subscription<std_msgs::msg::String>("/qrb_speaker", 10,
      std::bind(&SpeakerNode::handle_speech_request, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "SpeakerNode ready — subscribed to /qrb_speaker");
}

SpeakerNode::~SpeakerNode()
{
  synthesizer_.teardown();
}

void SpeakerNode::handle_speech_request(const std_msgs::msg::String::SharedPtr msg)
{
  if (!msg || msg->data.empty()) {
    RCLCPP_WARN(get_logger(), "Received empty speech request, ignoring");
    return;
  }

  RCLCPP_INFO(get_logger(), "Speaking: \"%s\"", msg->data.c_str());

  const Status st = synthesizer_.speak(msg->data.c_str(), static_cast<uint32_t>(msg->data.size()));
  if (st != Status::Ok) {
    RCLCPP_ERROR(get_logger(), "speak() failed (code=%d)", static_cast<int>(st));
  }
}

RCLCPP_COMPONENTS_REGISTER_NODE(SpeakerNode)
