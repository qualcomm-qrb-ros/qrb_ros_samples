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
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


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
    
    #qrb ros camera
    camera_info_config_file_path =PathJoinSubstitution([
        get_package_share_directory('qrb_ros_camera'),
        'config', 'camera_info_imx577.yaml'])
    camera_node = ComposableNode(
        package='qrb_ros_camera',
        namespace=namespace,
        plugin='qrb_ros::camera::CameraNode',
        name='camera_node',
        parameters=[{
            'camera_id': 0,
            'stream_size': 1,
            'stream_name': ["stream1"],
            'stream1':{
                'height':480,
                'width':640,
                'fps':30,
            },
            'camera_info_path': camera_info_config_file_path,
            'dump': False,
            'dump_camera_info_': False,
        }]
    )   
    
    preprocess_node = Node(
        package='sample_resnet101',
        executable='qrb_ros_resnet101',
        namespace=namespace,
        output='screen',  # Output logs to terminal
        remappings = [
            ("image_raw", "/cam0_stream1"),
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
        composable_node_descriptions = [nn_inference_node,camera_node]
    )
    
    return launch.LaunchDescription(declared_args + [preprocess_node, container,postprocess_node])
