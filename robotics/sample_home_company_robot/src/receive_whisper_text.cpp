#include "sample_home_company_robot/receive_whisper_text.hpp"

ReceiveWhisperText::ReceiveWhisperText(const std::string & name,
                                       const BT::NodeConfig & config,
                                       rclcpp::Node::SharedPtr node)
  : BT::StatefulActionNode(name, config), node_(node)
{
  sub_ = node_->create_subscription<std_msgs::msg::String>(
    "/whisper_out", 10,
    [this](const std_msgs::msg::String::SharedPtr msg) {
      if (!msg->data.empty()) {
        latest_text_  = msg->data;
        received_     = true;
      }
    });
}

BT::PortsList ReceiveWhisperText::providedPorts()
{
  return { BT::OutputPort<std::string>("whisper_text") };
}

BT::NodeStatus ReceiveWhisperText::onStart()
{
  received_     = false;
  latest_text_  = "";
  return onRunning();
}

BT::NodeStatus ReceiveWhisperText::onRunning()
{
  rclcpp::spin_some(node_);
  if (received_) {
    setOutput("whisper_text", latest_text_);
    RCLCPP_INFO(node_->get_logger(),
                "[ReceiveWhisperText] Received: \"%s\"", latest_text_.c_str());
    return BT::NodeStatus::SUCCESS;
  }
  return BT::NodeStatus::RUNNING;
}
