# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <vector>
#include <string>
#include <map>
#include <cmath> // For M_PI
 
/**
 * @brief The main function that starts our program.
 *
 * This function sets up our ROS 2 environment and prepares it for robot control.
 *
 * @param argc The number of input arguments our program receives.
 * @param argv The list of input arguments our program receives.
 * @return int A number indicating if our program finished successfully (0) or not.
 */
int main(int argc, char * argv[])
{
  // Start up ROS 2
  rclcpp::init(argc, argv);
 
  // Creates a node named "qrb_ros_arm_pick_place". The node is set up to automatically
  // handle any settings (parameters) we might want to change later without editing the code.
  auto const node = std::make_shared<rclcpp::Node>(
    "qrb_ros_arm_pick_place",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );
 
  // Creates a "logger" that we can use to print out information or error messages
  // as our program runs.
  auto const logger = rclcpp::get_logger("qrb_ros_arm_pick_place");
 
  // Create the MoveIt MoveGroup Interfaces for arm and gripper
  // These interfaces are used to plan and execute movements, set target poses,
  // and perform other motion-related tasks for each respective part of the robot.
  using moveit::planning_interface::MoveGroupInterface;
  auto arm_group_interface = MoveGroupInterface(node, "rm_group_controller");
  auto gripper_group_interface = MoveGroupInterface(node, "hand_controller");
 
  // Specify a planning pipeline to be used for further planning
  arm_group_interface.setPlanningPipelineId("ompl");
  gripper_group_interface.setPlanningPipelineId("ompl");
 
  // Specify a planner to be used for further planning
  arm_group_interface.setPlannerId("RRTConnectkConfigDefault");
  gripper_group_interface.setPlannerId("RRTConnectkConfigDefault");
 
  // Specify the maximum amount of time in seconds to use when planning
  arm_group_interface.setPlanningTime(1.0);
  gripper_group_interface.setPlanningTime(1.0);
 
  // Set a scaling factor for optionally reducing the maximum joint velocity. Allowed values are in (0,1].
  arm_group_interface.setMaxVelocityScalingFactor(0.3);
  gripper_group_interface.setMaxVelocityScalingFactor(0.3);
 
  //  Set a scaling factor for optionally reducing the maximum joint acceleration. Allowed values are in (0,1].
  arm_group_interface.setMaxAccelerationScalingFactor(0.3);
  gripper_group_interface.setMaxAccelerationScalingFactor(0.3);
 
  // Display helpful logging messages on the terminal
  RCLCPP_INFO(logger, "Planning pipeline: %s", arm_group_interface.getPlanningPipelineId().c_str());
  RCLCPP_INFO(logger, "Planner ID: %s", arm_group_interface.getPlannerId().c_str());
  RCLCPP_INFO(logger, "Planning time: %.2f", arm_group_interface.getPlanningTime());
 
  // Step 1: Move to ready state
  RCLCPP_INFO(logger, "Step 1: Moving to ready state...");
  arm_group_interface.setNamedTarget("ready");
  
  auto const [ready_success, ready_plan] = [&arm_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(arm_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();
 
  if (ready_success) {
    RCLCPP_INFO(logger, "Ready state planning successful! Executing...");
    arm_group_interface.execute(ready_plan);
    RCLCPP_INFO(logger, "Ready state movement completed!");
  } else {
    RCLCPP_ERROR(logger, "Ready state planning failed!");
    rclcpp::shutdown();
    return 1;
  }
 
  // Step 2: Open gripper
  RCLCPP_INFO(logger, "Step 2: Opening gripper...");
  
  // First reset gripper to a safe state to avoid joint limit violations
  RCLCPP_INFO(logger, "Resetting gripper to safe state...");
  gripper_group_interface.setNamedTarget("open");
  
  auto const [reset_success, reset_plan] = [&gripper_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(gripper_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();
 
  if (reset_success) {
    RCLCPP_INFO(logger, "Gripper reset planning successful! Executing...");
    gripper_group_interface.execute(reset_plan);
    RCLCPP_INFO(logger, "Gripper reset completed!");
  } else {
    RCLCPP_ERROR(logger, "Gripper reset planning failed!");
    rclcpp::shutdown();
    return 1;
  }
  
  gripper_group_interface.setNamedTarget("open");
  
  auto const [open_success, open_plan] = [&gripper_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(gripper_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();
 
  if (open_success) {
    RCLCPP_INFO(logger, "Gripper open planning successful! Executing...");
    gripper_group_interface.execute(open_plan);
    RCLCPP_INFO(logger, "Gripper opened successfully!");
  } else {
    RCLCPP_ERROR(logger, "Gripper open planning failed!");
    rclcpp::shutdown();
    return 1;
  }
 
  // Step 3: Move to target joint angles
  RCLCPP_INFO(logger, "Step 3: Moving to target joint angles...");
  
  // Get joint names
  std::vector<std::string> joint_names = arm_group_interface.getJointNames();
  RCLCPP_INFO(logger, "Available joints:");
  for (const auto& joint_name : joint_names) {
    RCLCPP_INFO(logger, "  %s", joint_name.c_str());
  }
 
  // Set target joint angles
  std::map<std::string, double> target_joint_values;
  target_joint_values["joint1"] = 0.0;    // 0°
  target_joint_values["joint2"] = -46.0;  // -46°
  target_joint_values["joint3"] = -114.0; // -114°
  target_joint_values["joint4"] = 0.0;    // 0°
  target_joint_values["joint5"] = 74.0;   // 74°
  target_joint_values["joint6"] = 0.0;    // 0°
 
  // Convert degrees to radians
  for (auto& [joint_name, angle_deg] : target_joint_values) {
    double angle_rad = angle_deg * M_PI / 180.0;
    target_joint_values[joint_name] = angle_rad;
    RCLCPP_INFO(logger, "Target %s: %.2f° (%.4f rad)", joint_name.c_str(), angle_deg, angle_rad);
  }
 
  // Set target joint angles
  arm_group_interface.setJointValueTarget(target_joint_values);
 
  auto const [target_success, target_plan] = [&arm_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(arm_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();
 
  if (target_success) {
    RCLCPP_INFO(logger, "Target joint angles planning successful! Executing...");
    arm_group_interface.execute(target_plan);
    RCLCPP_INFO(logger, "Target joint angles movement completed!");
  } else {
    RCLCPP_ERROR(logger, "Target joint angles planning failed!");
    rclcpp::shutdown();
    return 1;
  }
 
  // Step 4: Close gripper
  RCLCPP_INFO(logger, "Step 4: Closing gripper...");
  
  // Reset gripper to safe state before closing
  RCLCPP_INFO(logger, "Resetting gripper to safe state before closing...");
  gripper_group_interface.setNamedTarget("open");
  
  auto const [reset_close_success, reset_close_plan] = [&gripper_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(gripper_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();
 
  if (reset_close_success) {
    RCLCPP_INFO(logger, "Gripper reset before close planning successful! Executing...");
    gripper_group_interface.execute(reset_close_plan);
    RCLCPP_INFO(logger, "Gripper reset before close completed!");
  } else {
    RCLCPP_ERROR(logger, "Gripper reset before close planning failed!");
    rclcpp::shutdown();
    return 1;
  }
  
  gripper_group_interface.setNamedTarget("close");
  
  auto const [close_success, close_plan] = [&gripper_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(gripper_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();
 
  if (close_success) {
    RCLCPP_INFO(logger, "Gripper close planning successful! Executing...");
    gripper_group_interface.execute(close_plan);
    RCLCPP_INFO(logger, "Gripper closed successfully!");
  } else {
    RCLCPP_ERROR(logger, "Gripper close planning failed!");
    rclcpp::shutdown();
    return 1;
  }

  // Step 5: Move to new target joint angles
  RCLCPP_INFO(logger, "Step 5: Moving to new target joint angles...");

  // Set new target joint angles
  std::map<std::string, double> raise_joint_values;
  raise_joint_values["joint1"] = -22.0; 
  raise_joint_values["joint2"] = -37.0; 
  raise_joint_values["joint3"] = -48.0; 
  raise_joint_values["joint4"] = 0.0;  
  raise_joint_values["joint5"] = 0.0; 
  raise_joint_values["joint6"] = 0.0; 

  // Convert degrees to radians
  for (auto& [joint_name, angle_deg] : raise_joint_values) {
    double angle_rad = angle_deg * M_PI / 180.0;
    raise_joint_values[joint_name] = angle_rad;
    RCLCPP_INFO(logger, "New target %s: %.2f° (%.4f rad)", joint_name.c_str(), angle_deg, angle_rad);
  }

  // Set new target joint angles
  arm_group_interface.setJointValueTarget(raise_joint_values);

  auto const [raise_target_success, raise_target_plan] = [&arm_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(arm_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  if (raise_target_success) {
    RCLCPP_INFO(logger, "New target joint angles planning successful! Executing...");
    arm_group_interface.execute(raise_target_plan);
    RCLCPP_INFO(logger, "New target joint angles movement completed!");
  } else {
    RCLCPP_ERROR(logger, "New target joint angles planning failed!");
    rclcpp::shutdown();
    return 1;
  }
  
  // Step 6: Move to new target joint angles
  RCLCPP_INFO(logger, "Step 6: Moving to new target joint angles...");
  
  // Set new target joint angles
  std::map<std::string, double> new_target_joint_values;
  new_target_joint_values["joint1"] = -31.0; 
  new_target_joint_values["joint2"] = -42.0; 
  new_target_joint_values["joint3"] = -59.0; 
  new_target_joint_values["joint4"] = 0.0; 
  new_target_joint_values["joint5"] = 0.0;  
  new_target_joint_values["joint6"] = 0.0; 
 
  // Convert degrees to radians
  for (auto& [joint_name, angle_deg] : new_target_joint_values) {
    double angle_rad = angle_deg * M_PI / 180.0;
    new_target_joint_values[joint_name] = angle_rad;
    RCLCPP_INFO(logger, "New target %s: %.2f° (%.4f rad)", joint_name.c_str(), angle_deg, angle_rad);
  }
 
  // Set new target joint angles
  arm_group_interface.setJointValueTarget(new_target_joint_values);
 
  auto const [new_target_success, new_target_plan] = [&arm_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(arm_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();
 
  if (new_target_success) {
    RCLCPP_INFO(logger, "New target joint angles planning successful! Executing...");
    arm_group_interface.execute(new_target_plan);
    RCLCPP_INFO(logger, "New target joint angles movement completed!");
  } else {
    RCLCPP_ERROR(logger, "New target joint angles planning failed!");
    rclcpp::shutdown();
    return 1;
  }
 
  // Step 7: Open gripper (release)
  RCLCPP_INFO(logger, "Step 7: Opening gripper (releasing)...");
  
  // Reset gripper to safe state before releasing
  RCLCPP_INFO(logger, "Resetting gripper to safe state before releasing...");
  gripper_group_interface.setNamedTarget("open");
  
  auto const [reset_release_success, reset_release_plan] = [&gripper_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(gripper_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();
 
  if (reset_release_success) {
    RCLCPP_INFO(logger, "Gripper reset before release planning successful! Executing...");
    gripper_group_interface.execute(reset_release_plan);
    RCLCPP_INFO(logger, "Gripper reset before release completed!");
  } else {
    RCLCPP_ERROR(logger, "Gripper reset before release planning failed!");
    rclcpp::shutdown();
    return 1;
  }
  
  gripper_group_interface.setNamedTarget("open");
  
  auto const [release_success, release_plan] = [&gripper_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(gripper_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();
 
  if (release_success) {
    RCLCPP_INFO(logger, "Gripper release planning successful! Executing...");
    gripper_group_interface.execute(release_plan);
    RCLCPP_INFO(logger, "Gripper released successfully!");
  } else {
    RCLCPP_ERROR(logger, "Gripper release planning failed!");
    rclcpp::shutdown();
    return 1;
  }
 
 // Step 8: Move to new target joint angles (from the new image)
  RCLCPP_INFO(logger, "Step 8: Moving to new target joint angles...");

  // Set new target joint angles (from the new image)
  std::map<std::string, double> re_raise_joint_values;
  re_raise_joint_values["joint1"] = -22.0; 
  re_raise_joint_values["joint2"] = -35.0; 
  re_raise_joint_values["joint3"] = -52.0; 
  re_raise_joint_values["joint4"] = 0.0;
  re_raise_joint_values["joint5"] = 0.0; 
  re_raise_joint_values["joint6"] = 0.0;  

  // Convert degrees to radians
  for (auto& [joint_name, angle_deg] : re_raise_joint_values) {
    double angle_rad = angle_deg * M_PI / 180.0;
    re_raise_joint_values[joint_name] = angle_rad;
    RCLCPP_INFO(logger, "New target %s: %.2f° (%.4f rad)", joint_name.c_str(), angle_deg, angle_rad);
  }

  // Set new target joint angles
  arm_group_interface.setJointValueTarget(re_raise_joint_values);

  auto const [re_raise_target_success, re_raise_target_plan] = [&arm_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(arm_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  if (re_raise_target_success) {
    RCLCPP_INFO(logger, "New target joint angles planning successful! Executing...");
    arm_group_interface.execute(re_raise_target_plan);
    RCLCPP_INFO(logger, "New target joint angles movement completed!");
  } else {
    RCLCPP_ERROR(logger, "New target joint angles planning failed!");
    rclcpp::shutdown();
    return 1;
  }
 
  // Step 9: Return to ready state
  RCLCPP_INFO(logger, "Step 9: Returning to ready state...");
  arm_group_interface.setNamedTarget("ready");
  
  auto const [return_ready_success, return_ready_plan] = [&arm_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(arm_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();
  
  if (return_ready_success) {
    RCLCPP_INFO(logger, "Return to ready state planning successful! Executing...");
    arm_group_interface.execute(return_ready_plan);
    RCLCPP_INFO(logger, "Return to ready state completed!");
  } else {
    RCLCPP_ERROR(logger, "Return to ready state planning failed!");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(logger, "All steps completed successfully!");
 
  // Shut down ROS 2 cleanly when we're done
  rclcpp::shutdown();
  return 0;
}