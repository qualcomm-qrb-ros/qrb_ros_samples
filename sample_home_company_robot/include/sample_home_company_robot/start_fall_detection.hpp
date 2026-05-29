#pragma once
#include <behaviortree_cpp/action_node.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

class StartFallDetection : public BT::SyncActionNode
{
public:
  StartFallDetection(const std::string & name, const BT::NodeConfig & config,
                     rclcpp::Node::SharedPtr node);

  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_;
};
