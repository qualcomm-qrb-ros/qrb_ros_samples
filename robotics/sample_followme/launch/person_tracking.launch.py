# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directory
    pkg_dir = get_package_share_directory('follow_me')
    
    # Configuration file path
    config_file = os.path.join(pkg_dir, 'config', 'tracking_params.yaml')
    
    # Declare launch arguments
    declare_namespace = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for the node'
    )
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    declare_log_level = DeclareLaunchArgument(
        'log_level',
        default_value='INFO',
        description='Log level (DEBUG, INFO, WARN, ERROR, FATAL)'
    )
    
    # Person Tracker node
    person_tracker_node = Node(
        package='follow_me',
        executable='person_tracker_node',
        name='person_tracker_node',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[
            config_file,
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }
        ],
        arguments=['--ros-args', '--log-level', ['person_tracker_node:=', LaunchConfiguration('log_level')]]
    )
    
    return LaunchDescription([
        declare_namespace,
        declare_use_sim_time,
        declare_log_level,
        person_tracker_node,
    ])
