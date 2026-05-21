#include "sample_home_company_robot/send_follow_me_service.hpp"

#include "follow_me/srv/state_control.hpp"

static constexpr uint8_t CMD_START = 0x01;

SendFollowMeService::SendFollowMeService(const std::string &name,
                                         const BT::NodeConfig &config,
                                         rclcpp::Node::SharedPtr node)
  : BT::StatefulActionNode(name, config), node_(node)
{
  // 修改 service 类型 + 名字
  client_ = node_->create_client<follow_me::srv::StateControl>("/follow_me/state_control");
}

BT::NodeStatus SendFollowMeService::onStart()
{
  if (!client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_WARN(node_->get_logger(), "[SendFollowMeService] Service not available");
  }

  // 修改 request 类型
  auto req = std::make_shared<follow_me::srv::StateControl::Request>();

  // 映射原逻辑：START -> set_state = 1
  req->set_state = CMD_START;

  future_       = client_->async_send_request(req).share();
  request_sent_ = true;

  RCLCPP_INFO(node_->get_logger(), "[SendFollowMeService] Sent state set to 1");

  return BT::NodeStatus::RUNNING;
}

BT::NodeStatus SendFollowMeService::onRunning()
{
  rclcpp::spin_some(node_);

  if (future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
    auto resp = future_.get();

    // 假设新 service 仍然有 success 字段（一般是这样）
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
