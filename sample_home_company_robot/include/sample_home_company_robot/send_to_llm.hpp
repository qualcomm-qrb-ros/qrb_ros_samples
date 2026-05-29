#pragma once
#include <behaviortree_cpp/action_node.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class SendToLLM : public BT::StatefulActionNode
{
public:
  SendToLLM(const std::string & name, const BT::NodeConfig & config,
            rclcpp::Node::SharedPtr node);

  static BT::PortsList providedPorts();
  BT::NodeStatus onStart()   override;
  BT::NodeStatus onRunning() override;
  void           onHalted()  override {}

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr  pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
  std::string llm_reply_;
  bool reply_received_{false};
};
