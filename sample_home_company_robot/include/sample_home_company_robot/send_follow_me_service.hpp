#pragma once
#include <behaviortree_cpp/action_node.h>
#include <rclcpp/rclcpp.hpp>
#include "sample_home_company_robot/send_follow_me_service.hpp"
#include "follow_me/srv/state_control.hpp"

class SendFollowMeService : public BT::StatefulActionNode
{
public:
  SendFollowMeService(const std::string & name, const BT::NodeConfig & config,
                      rclcpp::Node::SharedPtr node);

  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus onStart()   override;
  BT::NodeStatus onRunning() override;
  void           onHalted()  override {}

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Client<follow_me::srv::StateControl>::SharedPtr client_;
  rclcpp::Client<follow_me::srv::StateControl>::SharedFuture future_;
  bool request_sent_{false};
};
