# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

#!/usr/bin/env python3
"""
Simplified demo launch file for RViz and Gazebo simulation.
This launch file focuses on the essential components for visualization and simulation.
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for simplified RViz and Gazebo demo."""
    
    # Declare launch arguments
    ld = LaunchDescription()
    
    ld.add_action(DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'use_gazebo',
        default_value='true',
        description='Whether to start Gazebo simulation'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'world_model',
        default_value='pick_and_place_demo',
        description='Gazebo world model to load'
    ))
    
    # Launch Gazebo simulation
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('qrb_ros_sim_gazebo'),
                'launch',
                'gazebo_rml_63_gripper.launch.py'
            )
        ]),
        launch_arguments={
            'world_model': LaunchConfiguration('world_model'),
            'enable_rgb_camera': 'false',
            'enable_depth_camera': 'false',
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_gazebo'))
    )
    
    # Launch MoveIt components
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('qrb_ros_arm_moveit_config'),
                'launch',
                'move_group.launch.py'
            )
        ])
    )
    
    # Launch RViz
    rviz_config_file = os.path.join(
        get_package_share_directory('qrb_ros_arm_moveit_config'),
        'rviz',
        'moveit.rviz'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )
    
    # Add actions with proper timing
    ld.add_action(gazebo_launch)
    ld.add_action(TimerAction(
        period=5.0,  # Wait 5 seconds for Gazebo to start
        actions=[moveit_launch]
    ))
    ld.add_action(TimerAction(
        period=7.0,  # Wait 7 seconds for everything to be ready
        actions=[rviz_node]
    ))
    
    return ld 