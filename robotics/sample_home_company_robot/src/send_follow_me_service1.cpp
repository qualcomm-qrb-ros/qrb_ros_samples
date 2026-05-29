#include "sample_home_company_robot/send_follow_me_service.hpp"

static constexpr uint8_t CMD_START = 0x01;

SendFollowMeService::SendFollowMeService(const std::string & name,
                                         const BT::NodeConfig & config,
                                         rclcpp::Node::SharedPtr node)
  : BT::StatefulActionNode(name, config), node_(node)
{
  client_ = node_->create_client<sample_home_company_robot::srv::FollowMeStart>("/follow_me_start");
}

BT::NodeStatus SendFollowMeService::onStart()
{
  if (!client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_WARN(node_->get_logger(), "[SendFollowMeService] Service not available");
    //return BT::NodeStatus::FAILURE;
    //return BT::NodeStatus::RUNNING;
  }

  auto req = std::make_shared<sample_home_company_robot::srv::FollowMeStart::Request>();
  req->command = CMD_START;
  future_       = client_->async_send_request(req).share();
  request_sent_ = true;
  RCLCPP_INFO(node_->get_logger(), "[SendFollowMeService] Sent command 0x01");
  return BT::NodeStatus::RUNNING;
}

BT::NodeStatus SendFollowMeService::onRunning()
{
  rclcpp::spin_some(node_);
  if (future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
    auto resp = future_.get();
    if (resp->success) {
      RCLCPP_INFO(node_->get_logger(), "[SendFollowMeService] Service returned success=true");
      return BT::NodeStatus::SUCCESS;
    } else {
      RCLCPP_WARN(node_->get_logger(), "[SendFollowMeService] Service returned success=false");
      return BT::NodeStatus::FAILURE;
    }
  }
  return BT::NodeStatus::RUNNING;
}
