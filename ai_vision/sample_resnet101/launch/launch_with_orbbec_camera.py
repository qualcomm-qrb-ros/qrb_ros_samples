# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os  # For accessing environment variables
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.logging import get_logger
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    
    declared_args = [
        DeclareLaunchArgument(
            'model_path',
            default_value="/opt/model/ResNet101_w8a8.bin",
            description='Path to the model file'
        ),
    ]

    model_path = LaunchConfiguration('model_path')
    
    namespace = ""
    
    #orbbec camera
    another_launch_path = os.path.join(
        get_package_share_directory('orbbec_camera'),
        'launch',
        'gemini_330_series.launch.py'
    )
    included_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(another_launch_path),
        launch_arguments={
            'color_width': '640',
            'color_height': '480',
            'color_fps': '30',
            'color_qos': 'default'
        }.items()
    )
    
    preprocess_node = Node(
        package='sample_resnet101',
        executable='qrb_ros_resnet101',
        namespace=namespace,
        output='screen',  # Output logs to terminal
        remappings = [
            ("image_raw", "/camera/color/image_raw"),
        ],
    )
    
    nn_inference_node = ComposableNode(
    package = "qrb_ros_nn_inference",
    namespace=namespace,
    plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
    name = "nn_inference_node",
    parameters = [
      {
        "backend_option": "/usr/lib/libQnnHtp.so",
        "model_path": model_path
      }]
    )
    
    postprocess_node = Node(
        package='sample_resnet101',
        executable='qrb_ros_resnet101_posprocess',
        namespace=namespace,
        output='screen',
	)
    
    container = ComposableNodeContainer(
        name = "container",
        namespace=namespace,
        package = "rclcpp_components",
        executable='component_container',
        output = "screen",
        composable_node_descriptions = [nn_inference_node]
    )
    
    return launch.LaunchDescription(declared_args + [preprocess_node, container,postprocess_node,included_launch])
