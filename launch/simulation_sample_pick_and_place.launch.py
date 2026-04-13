# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from moveit_configs_utils import MoveItConfigsBuilder

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils.launch_utils import add_debuggable_node


def generate_launch_description():
    # Keep the same config stem name used by this package (rml_63.*)
    moveit_config = MoveItConfigsBuilder(
        'rml_63', package_name='simulation_sample_pick_and_place'
    ).to_moveit_configs()

    ld = LaunchDescription()

    # Common args
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='true'))
    ld.add_action(DeclareLaunchArgument('debug', default_value='false'))
    ld.add_action(DeclareLaunchArgument('allow_trajectory_execution', default_value='true'))
    ld.add_action(DeclareLaunchArgument('publish_monitored_planning_scene', default_value='true'))
    ld.add_action(DeclareLaunchArgument('capabilities', default_value=''))
    ld.add_action(DeclareLaunchArgument('disable_capabilities', default_value=''))
    ld.add_action(DeclareLaunchArgument('publish_robot_description', default_value='false'))
    ld.add_action(DeclareLaunchArgument('publish_robot_description_semantic', default_value='false'))

    should_publish_scene = LaunchConfiguration('publish_monitored_planning_scene')

    # Host-collab mode: do not publish robot_description topics to avoid collisions
    move_group_configuration = {
        'allow_trajectory_execution': LaunchConfiguration('allow_trajectory_execution'),
        'capabilities': ParameterValue(LaunchConfiguration('capabilities'), value_type=str),
        'disable_capabilities': ParameterValue(
            LaunchConfiguration('disable_capabilities'), value_type=str
        ),
        'publish_planning_scene': should_publish_scene,
        'publish_geometry_updates': should_publish_scene,
        'publish_state_updates': should_publish_scene,
        'publish_transforms_updates': should_publish_scene,
        'publish_robot_description': LaunchConfiguration('publish_robot_description'),
        'publish_robot_description_semantic': LaunchConfiguration('publish_robot_description_semantic'),
        'monitor_dynamics': False,
        'use_sim_time': LaunchConfiguration('use_sim_time'),
    }

    trajectory_execution = {
        'moveit_manage_controllers': False,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.15,
    }

    move_group_params = [
        moveit_config.to_dict(),
        move_group_configuration,
        trajectory_execution,
    ]

    add_debuggable_node(
        ld,
        package='moveit_ros_move_group',
        executable='move_group',
        commands_file=str(moveit_config.package_path / 'launch' / 'gdb_settings.gdb'),
        output='screen',
        parameters=move_group_params,
        extra_debug_args=['--debug'],
        additional_env={'DISPLAY': ':0'},
    )

    # Launch pick-and-place app in the same graph with explicit local description params.
    ld.add_action(
        Node(
            package='simulation_sample_pick_and_place',
            executable='qrb_ros_arm_pick_place',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
            ],
        )
    )

    return ld
