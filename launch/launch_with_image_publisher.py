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
    logger = get_logger('image_publisher_launch')

    # Declare the launch arguments for image_path and model_path
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value=os.path.join(get_package_share_directory('sample_hrnet_pose_estimation'), 'input_image.jpg'),
        description='Path to the image file'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value= "/opt/model/",
        description='Path to the model file'
    )
    
    # Use LaunchConfiguration to get the values of the arguments
    image_path = LaunchConfiguration('image_path')
    model_path = LaunchConfiguration('model_path')

    logger.info(f'IMAGE_PATH set to: {image_path}')
    logger.info(f'MODEL_PATH set to: {model_path}')

    hr_pose_estimation_node = Node(
        package='sample_hrnet_pose_estimation',
        executable='sample_hrnet_pose_estimation',
        output='screen',
    )

    image_publisher_node = Node(
        package='image_publisher',
        executable='image_publisher_node',
        name='image_publisher_node',
        output='screen',
        parameters=[
            {'filename': image_path},
            {'rate': 10.0},
        ],
    )

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
                        "model_path": PathJoinSubstitution([model_path, "HRNetPose.bin"])
                    }
                ]
            )
        ]
    )

    return LaunchDescription([
        image_path_arg,
        model_path_arg,
        image_publisher_node,
        nn_inference_container,
        hr_pose_estimation_node
    ])