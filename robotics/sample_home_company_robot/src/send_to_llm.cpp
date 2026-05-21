#include "sample_home_company_robot/send_to_llm.hpp"

SendToLLM::SendToLLM(const std::string & name,
                     const BT::NodeConfig & config,
                     rclcpp::Node::SharedPtr node)
  : BT::StatefulActionNode(name, config), node_(node)
{
  pub_ = node_->create_publisher<std_msgs::msg::String>("/llm_input", 10);

  sub_ = node_->create_subscription<std_msgs::msg::String>(
    "/llm_output", 10,
    [this](const std_msgs::msg::String::SharedPtr msg) {
      if (!msg->data.empty()) {
        llm_reply_      = msg->data;
        reply_received_ = true;
      }
    });
}

BT::PortsList SendToLLM::providedPorts()
{
  return {
    BT::InputPort<std::string>("whisper_text"),
    BT::OutputPort<std::string>("llm_text")
  };
}

BT::NodeStatus SendToLLM::onStart()
{
  reply_received_ = false;
  llm_reply_      = "";

  std::string text;
  if (!getInput("whisper_text", text)) {
    RCLCPP_WARN(node_->get_logger(), "[SendToLLM] whisper_text not available");
    return BT::NodeStatus::FAILURE;
  }

  std_msgs::msg::String msg;
  msg.data = text;
  pub_->publish(msg);
  RCLCPP_INFO(node_->get_logger(),
              "[SendToLLM] Published to /llm_input: \"%s\"", text.c_str());
  return BT::NodeStatus::RUNNING;
}

BT::NodeStatus SendToLLM::onRunning()
{
  rclcpp::spin_some(node_);
  if (reply_received_) {
    setOutput("llm_text", llm_reply_);
    RCLCPP_INFO(node_->get_logger(),
                "[SendToLLM] LLM reply received: \"%s\"", llm_reply_.c_str());
    return BT::NodeStatus::SUCCESS;
  }
  return BT::NodeStatus::RUNNING;
}
