# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.logging import get_logger
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import LogInfo
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    logger = get_logger('lidar_semantic_segmentation_launch')

    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(get_package_share_directory('sample_3d_lidar_semantic_segmentation'), 'model'),
        description='Path to the model directory'
    )

    fov_up_arg = DeclareLaunchArgument(
        'fov_up',
        default_value='3.0',
        description='Vertical field of view upper limit (degrees)'
    )

    fov_down_arg = DeclareLaunchArgument(
        'fov_down',
        default_value='-25.0',
        description='Vertical field of view lower limit (degrees)'
    )
    
    # Use LaunchConfiguration to get the values of the arguments
    model_path = LaunchConfiguration('model_path')
    fov_up = LaunchConfiguration('fov_up')
    fov_down = LaunchConfiguration('fov_down')

    logger.info(f'MODEL_PATH set to: {model_path}')
    logger.info(f'FOV_UP set to: {fov_up}')
    logger.info(f'FOV_DOWN set to: {fov_down}')

    # Create semantic segmentation node
    lidar_semantic_segmentation_node = Node(
        package='sample_3d_lidar_semantic_segmentation',
        executable='lidar_semantic_segmentation_node',
        name='lidar_semantic_segmentation_node',
        output='screen',
        parameters=[
            {'fov_up': fov_up},
            {'fov_down': fov_down},
        ],
    )

    # Create nn inference container
    nn_inference_container = ComposableNodeContainer(
        name="container",
        namespace='',
        package="rclcpp_components",
        executable="component_container",
        output='screen',
        composable_node_descriptions=[
            ComposableNode(
                package = "qrb_ros_nn_inference",
                plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
                name = "nn_inference_node",
                parameters=[
                    {
                        "backend_option": "/usr/lib/libQnnHtp.so",
                        "model_path": PathJoinSubstitution([model_path, "libmodel.so"])
                    }
                ]
            )
        ]
    )

    return LaunchDescription([
        model_path_arg,
        fov_up_arg,
        fov_down_arg,
        nn_inference_container,
        lidar_semantic_segmentation_node
    ])