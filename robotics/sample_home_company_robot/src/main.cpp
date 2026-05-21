#include <rclcpp/rclcpp.hpp>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/loggers/bt_cout_logger.h>

#include "sample_home_company_robot/receive_whisper_text.hpp"
#include "sample_home_company_robot/check_follow_me.hpp"
#include "sample_home_company_robot/send_follow_me_service.hpp"
#include "sample_home_company_robot/start_fall_detection.hpp"
#include "sample_home_company_robot/send_to_llm.hpp"
#include "sample_home_company_robot/send_to_tts.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("bt_ros_node");

  BT::BehaviorTreeFactory factory;

  // Register all nodes with shared ROS node
  factory.registerNodeType<ReceiveWhisperText>("ReceiveWhisperText", node);
  factory.registerNodeType<CheckFollowMe>     ("CheckFollowMe",      node);
  factory.registerNodeType<SendFollowMeService>("SendFollowMeService", node);
  factory.registerNodeType<StartFallDetection>("StartFallDetection", node);
  factory.registerNodeType<SendToLLM>         ("SendToLLM",          node);
  factory.registerNodeType<SendToTTS>         ("SendToTTS",          node);

  // Load tree from XML
  auto tree = factory.createTreeFromFile(
    ament_index_cpp::get_package_share_directory("sample_home_company_robot") + "/config/tree.xml");

  BT::StdCoutLogger logger(tree);

  RCLCPP_INFO(node->get_logger(), "[BT] Starting behavior tree tick loop...");

  rclcpp::Rate rate(10);  // 10 Hz
  while (rclcpp::ok()) {
    rclcpp::spin_some(node);
    tree.tickOnce();
    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
