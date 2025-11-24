# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import LogInfo

def generate_launch_description():
    package_name = 'sample_depth_estimation' 
    package_path = get_package_share_directory(package_name)

    # Declare the launch arguments for image_path and model_path
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value=os.path.join(package_path, "resource", "input_image.jpg"),
        description='Path to the input image file'
    )
      
    image_path = LaunchConfiguration('image_path')
    LogInfo(msg=['IMAGE_PATH: ', image_path])

    model_path_arg = DeclareLaunchArgument(
       'model_path',
        default_value="/opt/model/Depth-Anything-V2.bin",
        description='Path to the model file'
    )
    model_path= LaunchConfiguration('model_path')
    LogInfo(msg=['MODEL_PATH: ', model_path])

    namespace = "sample_container"
    # Node for image_publisher
    image_publish_node = Node(        
        package='image_publisher', 
        executable='image_publisher_node', 
        name='image_publisher_node', 
        output='screen', 
        parameters=[
            {'filename': image_path}, 
            {'rate': 10.0},  # Set the publishing rate to 10 Hz
        ],
    )

    # Node for depth estimation
    depth_estimation_node = Node(
        package='sample_depth_estimation',
        executable='depth_estimation_node', 
        name='depth_estimation_node', 
        namespace=namespace, 
    )

    # Node fir qnn inference
    nn_inference_node = ComposableNode(
        package = "qrb_ros_nn_inference",
        namespace=namespace,
        plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
        name = "nn_inference_node",
        parameters = [
        {
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": model_path,
            "log_level": "warn"
        }]
    )

    container = ComposableNodeContainer(
        name = "container",
        namespace=namespace,
        package = "rclcpp_components",
        executable='component_container',
        output = "screen",
        composable_node_descriptions = [nn_inference_node],
        # Add sigterm timeout to allow graceful shutdown
        sigterm_timeout='3',
        sigkill_timeout='5'
    )

    return LaunchDescription(
        [
            image_path_arg,
            model_path_arg,
            image_publish_node, 
            container,
            depth_estimation_node
        ]
    )