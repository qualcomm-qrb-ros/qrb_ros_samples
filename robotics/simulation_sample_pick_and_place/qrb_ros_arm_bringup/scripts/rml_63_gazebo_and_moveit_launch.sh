# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

#!/bin/bash
# Single script to launch the rml_63 with Gazebo, RViz, and MoveIt 2

cleanup() {
    echo "Cleaning up..."
    sleep 5.0
    pkill -9 -f "ros2|gazebo|gz|nav2|amcl|bt_navigator|nav_to_pose|rviz2|assisted_teleop|cmd_vel_relay|robot_state_publisher|joint_state_publisher|move_to_free|mqtt|autodock|cliff_detection|moveit|move_group|basic_navigator"
}

# Set up cleanup trap
trap 'cleanup' SIGINT SIGTERM

echo "Launching Gazebo simulation..."
ros2 launch qrb_ros_sim_gazebo gazebo_rml_63_gripper.launch.py world_model:=panda_world initial_x:=0.0 initial_y:=0 initial_z:=1.025 initial_yaw:=0.0 initial_pitch:=0.0 initial_roll:=0.0 &

sleep 25
ros2 launch qrb_ros_arm_moveit_config demo.launch.py &

# Keep the script running until Ctrl+C
wait