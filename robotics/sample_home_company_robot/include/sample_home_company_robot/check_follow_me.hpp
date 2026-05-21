#pragma once
#include <behaviortree_cpp/condition_node.h>
#include <rclcpp/rclcpp.hpp>

class CheckFollowMe : public BT::ConditionNode
{
public:
  CheckFollowMe(const std::string & name, const BT::NodeConfig & config,
                rclcpp::Node::SharedPtr node);

  static BT::PortsList providedPorts();
  BT::NodeStatus tick() override;

private:
  rclcpp::Node::SharedPtr node_;
};
