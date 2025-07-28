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

def generate_launch_description():
    logger = get_logger('image_publisher_launch')
    package_name = 'sample_resnet101'  
    package_path = get_package_share_directory(package_name)
    #image_path = os.path.join(package_path, 'cup.jpg')  
    #image_path = os.path.join(package_path, 'glasses.jpg')
    #image_path = os.path.join(package_path, 'glasses.jpg')
  
    # Declare arguments
    declared_args = [
        DeclareLaunchArgument(
            'model_path',
            default_value="/opt/model/ResNet101_w8a8.bin",
            description='Path to the model file'
        ),
        DeclareLaunchArgument(
            'image_path',
            default_value=os.path.join(package_path, 'glasses.jpg'),
            description='Path to the image file'
        )
    ]
    
    image_path = LaunchConfiguration('image_path')
    model_path = LaunchConfiguration('model_path')
    
    namespace = ""
    
    # Node for image_publisher
    image_publisher_node = Node(
        package='image_publisher',  # Package name
        executable='image_publisher_node',  # Executable name
        namespace=namespace,
        name='image_publisher_node',  # Node name
        output='screen',  # Output logs to terminal
        parameters=[
            {'filename': image_path},  # Use the constructed relative path
            {'rate': 10.0},  # Set the publishing rate to 10 Hz
        ]
    )
    
    preprocess_node = Node(
        package='sample_resnet101',
        executable='qrb_ros_resnet101',
        namespace=namespace,
        output='screen',  # Output logs to terminal
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
      }
      ]
    )
    
    postprocess_node = Node(
        package='sample_resnet101',  # Replace with the actual package name
        executable='qrb_ros_resnet101_posprocess',  # Replace with the actual executable name
        namespace=namespace,
        output='screen',  # Output logs to terminal
    )
    
    container = ComposableNodeContainer(
        name = "container",
        namespace=namespace,
        package = "rclcpp_components",
        executable='component_container',
        output = "screen",
        composable_node_descriptions = [nn_inference_node]
    )
    
    return launch.LaunchDescription(declared_args + [image_publisher_node,preprocess_node, container,postprocess_node])
