#include "sample_home_company_robot/send_to_tts.hpp"

SendToTTS::SendToTTS(const std::string & name,
                     const BT::NodeConfig & config,
                     rclcpp::Node::SharedPtr node)
  : BT::SyncActionNode(name, config), node_(node)
{
  pub_ = node_->create_publisher<std_msgs::msg::String>("/tts_input", 10);
}

BT::PortsList SendToTTS::providedPorts()
{
  return { BT::InputPort<std::string>("llm_text") };
}

BT::NodeStatus SendToTTS::tick()
{
  std::string text;
  if (!getInput("llm_text", text)) {
    RCLCPP_WARN(node_->get_logger(), "[SendToTTS] llm_text not available");
    return BT::NodeStatus::FAILURE;
  }

  std_msgs::msg::String msg;
  msg.data = text;
  pub_->publish(msg);
  RCLCPP_INFO(node_->get_logger(),
              "[SendToTTS] Published to /tts_input: \"%s\"", text.c_str());
  return BT::NodeStatus::SUCCESS;
}
