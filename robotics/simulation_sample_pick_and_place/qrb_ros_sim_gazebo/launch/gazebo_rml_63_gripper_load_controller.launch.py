# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os
import yaml
import xacro
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    robot_gazebo_pkg = 'qrb_ros_sim_gazebo'
    gz_pkg_share = get_package_share_directory(robot_gazebo_pkg)

    launch_args = [
        DeclareLaunchArgument('namespace', default_value='')
    ]

    return LaunchDescription(launch_args + [
        OpaqueFunction(function=generate_robot_nodes)
    ])

def generate_robot_nodes(context):
    namespace = LaunchConfiguration('namespace').perform(context),

    load_joint_state_controller = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        arguments=[
            'joint_state_broadcaster',
            ],
    )

    load_joint_trajectory_controller = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        arguments=[
            'rm_group_controller',
            ],
    )
    load_gripper_controller = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        arguments=[
            'hand_controller',
            ],
    )

    close_evt1 = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_controller,
                on_exit=[load_joint_trajectory_controller],
            )
    )
    close_evt2 = RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_controller,
                on_exit=[load_gripper_controller],
            )
    )

    return [
        load_joint_state_controller,
        close_evt1,
        close_evt2,
    ]
