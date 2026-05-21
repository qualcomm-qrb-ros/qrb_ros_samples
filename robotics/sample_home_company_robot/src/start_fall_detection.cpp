#include "sample_home_company_robot/start_fall_detection.hpp"

StartFallDetection::StartFallDetection(const std::string & name,
                                       const BT::NodeConfig & config,
                                       rclcpp::Node::SharedPtr node)
  : BT::SyncActionNode(name, config), node_(node)
{
  pub_ = node_->create_publisher<std_msgs::msg::Bool>("/start_fall_detection", 10);
}

BT::NodeStatus StartFallDetection::tick()
{
  std_msgs::msg::Bool msg;
  msg.data = true;
  pub_->publish(msg);
  RCLCPP_INFO(node_->get_logger(), "[StartFallDetection] Published true to /start_fall_detection");
  return BT::NodeStatus::SUCCESS;
}
