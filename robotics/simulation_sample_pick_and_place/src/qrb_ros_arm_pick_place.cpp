// Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include <moveit/move_group_interface/move_group_interface.h>

#include <cmath>  // For M_PI
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <map>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>

/**
 * @brief Robot control class that encapsulates pick and place operations
 */
class RobotController
{
public:
  RobotController(std::shared_ptr<rclcpp::Node> node)
    : node_(node)
    , logger_(rclcpp::get_logger("qrb_ros_arm_pick_place"))
    , arm_group_(node, "rm_group_controller")
    , gripper_group_(node, "hand_controller")
  {
    setupMoveGroups();
  }

  /**
   * @brief Execute the complete pick and place task
   */
  bool executePickAndPlace()
  {
    try {
      RCLCPP_INFO(logger_, "Starting pick and place task...");

      // Step 1: Move to ready state
      if (!moveToNamedTarget(arm_group_, "ready", "Ready State")) {
        return false;
      }

      // Step 2: Open gripper
      if (!moveToNamedTarget(gripper_group_, "open", "Open Gripper")) {
        return false;
      }

      // Step 3: Move to target joint angles (pick position)
      if (!moveToJointAngles(getPickJointAngles(), "Pick Position")) {
        return false;
      }

      // Step 4: Close gripper (grasp)
      if (!moveToNamedTarget(gripper_group_, "close", "Close Gripper")) {
        return false;
      }

      // Step 5: Lift object
      if (!moveToJointAngles(getRaiseJointAngles(), "Lift Object")) {
        return false;
      }

      // Step 6: Move to place position
      if (!moveToJointAngles(getPlaceJointAngles(), "Place Position")) {
        return false;
      }

      // Step 7: Open gripper (release)
      if (!moveToNamedTarget(gripper_group_, "open", "Release Object")) {
        return false;
      }

      // Step 8: Lift again
      if (!moveToJointAngles(getReRaiseJointAngles(), "Lift Again")) {
        return false;
      }

      // Step 9: Return to ready state
      if (!moveToNamedTarget(arm_group_, "ready", "Return to Ready State")) {
        return false;
      }

      RCLCPP_INFO(logger_, "Pick and place task completed!");
      return true;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(logger_, "Error during execution: %s", e.what());
      return false;
    }
  }

private:
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Logger logger_;
  moveit::planning_interface::MoveGroupInterface arm_group_;
  moveit::planning_interface::MoveGroupInterface gripper_group_;

  /**
   * @brief Setup common parameters for MoveGroup interfaces
   */
  void setupMoveGroups()
  {
    // Set planning pipeline and planner
    const std::string planning_pipeline = "ompl";
    const std::string planner_id = "RRTConnectkConfigDefault";
    const double planning_time = 3.0;
    const double velocity_scaling = 0.3;
    const double acceleration_scaling = 0.3;

    for (auto & group : { std::ref(arm_group_), std::ref(gripper_group_) }) {
      group.get().setPlanningPipelineId(planning_pipeline);
      group.get().setPlannerId(planner_id);
      group.get().setPlanningTime(planning_time);
      group.get().setMaxVelocityScalingFactor(velocity_scaling);
      group.get().setMaxAccelerationScalingFactor(acceleration_scaling);
    }

    RCLCPP_INFO(logger_, "Planning pipeline: %s", planning_pipeline.c_str());
    RCLCPP_INFO(logger_, "Planner ID: %s", planner_id.c_str());
    RCLCPP_INFO(logger_, "Planning time: %.2f seconds", planning_time);
  }

  /**
   * @brief Move to a named target
   */
  bool moveToNamedTarget(moveit::planning_interface::MoveGroupInterface & group,
      const std::string & target_name,
      const std::string & description)
  {
    RCLCPP_INFO(logger_, "Step: %s", description.c_str());

    group.setNamedTarget(target_name);

    auto [success, plan] = planAndExecute(group);
    if (success) {
      RCLCPP_INFO(logger_, "%s completed", description.c_str());
      return true;
    } else {
      RCLCPP_ERROR(logger_, "%s failed", description.c_str());
      return false;
    }
  }

  /**
   * @brief Move to specified joint angles
   */
  bool moveToJointAngles(const std::map<std::string, double> & joint_angles_deg,
      const std::string & description)
  {
    RCLCPP_INFO(logger_, "Step: %s", description.c_str());

    // Convert degrees to radians
    std::map<std::string, double> joint_angles_rad = convertToRadians(joint_angles_deg);

    // Print target angles
    for (const auto & [joint_name, angle_deg] : joint_angles_deg) {
      RCLCPP_INFO(logger_, "Target %s: %.2f° (%.4f rad)", joint_name.c_str(), angle_deg,
          joint_angles_rad[joint_name]);
    }

    arm_group_.setJointValueTarget(joint_angles_rad);

    auto [success, plan] = planAndExecute(arm_group_);
    if (success) {
      RCLCPP_INFO(logger_, "%s completed", description.c_str());
      return true;
    } else {
      RCLCPP_ERROR(logger_, "%s failed", description.c_str());
      return false;
    }
  }

  /**
   * @brief Plan and execute motion
   */
  std::pair<bool, moveit::planning_interface::MoveGroupInterface::Plan> planAndExecute(
      moveit::planning_interface::MoveGroupInterface & group)
  {
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(group.plan(plan));

    if (success) {
      RCLCPP_INFO(logger_, "Planning successful! Executing...");
      group.execute(plan);
    }

    return { success, plan };
  }

  /**
   * @brief Convert angles from degrees to radians
   */
  std::map<std::string, double> convertToRadians(const std::map<std::string, double> & angles_deg)
  {
    std::map<std::string, double> angles_rad;
    for (const auto & [joint_name, angle_deg] : angles_deg) {
      angles_rad[joint_name] = angle_deg * M_PI / 180.0;
    }
    return angles_rad;
  }

  /**
   * @brief Get joint angles for pick position
   */
  std::map<std::string, double> getPickJointAngles()
  {
    return {
      { "joint1", 0.0 },     // 0°
      { "joint2", -46.0 },   // -46°
      { "joint3", -114.0 },  // -114°
      { "joint4", 0.0 },     // 0°
      { "joint5", 74.0 },    // 74°
      { "joint6", 0.0 }      // 0°
    };
  }

  /**
   * @brief Get joint angles for lift position
   */
  std::map<std::string, double> getRaiseJointAngles()
  {
    return {
      { "joint1", -22.0 },  // -22°
      { "joint2", -37.0 },  // -37°
      { "joint3", -48.0 },  // -48°
      { "joint4", 0.0 },    // 0°
      { "joint5", 0.0 },    // 0°
      { "joint6", 0.0 }     // 0°
    };
  }

  /**
   * @brief Get joint angles for place position
   */
  std::map<std::string, double> getPlaceJointAngles()
  {
    return {
      { "joint1", -31.0 },  // -31°
      { "joint2", -42.0 },  // -42°
      { "joint3", -59.0 },  // -59°
      { "joint4", 0.0 },    // 0°
      { "joint5", 0.0 },    // 0°
      { "joint6", 0.0 }     // 0°
    };
  }

  /**
   * @brief Get joint angles for re-lift position
   */
  std::map<std::string, double> getReRaiseJointAngles()
  {
    return {
      { "joint1", -22.0 },  // -22°
      { "joint2", -35.0 },  // -35°
      { "joint3", -52.0 },  // -52°
      { "joint4", 0.0 },    // 0°
      { "joint5", 0.0 },    // 0°
      { "joint6", 0.0 }     // 0°
    };
  }
};

/**
 * @brief Main function
 */
int main(int argc, char * argv[])
{
  // Initialize ROS 2
  rclcpp::init(argc, argv);

  // Create node
  auto node = std::make_shared<rclcpp::Node>("qrb_ros_arm_pick_place",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  // Create robot controller and execute task
  RobotController controller(node);
  bool success = controller.executePickAndPlace();

  // Shutdown ROS 2
  rclcpp::shutdown();

  return success ? 0 : 1;
}
