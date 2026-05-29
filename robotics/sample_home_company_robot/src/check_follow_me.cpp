#include "sample_home_company_robot/check_follow_me.hpp"
#include <algorithm>

CheckFollowMe::CheckFollowMe(const std::string & name,
                             const BT::NodeConfig & config,
                             rclcpp::Node::SharedPtr node)
  : BT::ConditionNode(name, config), node_(node) {}

BT::PortsList CheckFollowMe::providedPorts()
{
  return { BT::InputPort<std::string>("whisper_text") };
}

BT::NodeStatus CheckFollowMe::tick()
{
  std::string text;
  if (!getInput("whisper_text", text)) {
    RCLCPP_WARN(node_->get_logger(), "[CheckFollowMe] whisper_text not available");
    return BT::NodeStatus::FAILURE;
  }

  // case-insensitive search
  std::string lower = text;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  if (lower.find("follow me") != std::string::npos) {
    RCLCPP_INFO(node_->get_logger(),
                "[CheckFollowMe] 'follow me' found in: \"%s\"", text.c_str());
    return BT::NodeStatus::SUCCESS;
  }

  RCLCPP_INFO(node_->get_logger(),
              "[CheckFollowMe] 'follow me' NOT found in: \"%s\"", text.c_str());
  return BT::NodeStatus::FAILURE;
}
