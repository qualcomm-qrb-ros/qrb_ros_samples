# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Declare the launch arguments for image_path and model_path
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value=os.path.join(get_package_share_directory('sample_hand_detection'), 'input_image.jpg'),
        description='Path to the image file'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/opt/model/',
        description='Path to the model file'
    )
    
    # Use LaunchConfiguration to get the values of the arguments
    image_path = LaunchConfiguration('image_path')
    model_path = LaunchConfiguration('model_path')

    # Node for palm detector
    nn_inference_node_palm_detector = ComposableNode(
        package = "qrb_ros_nn_inference",
        plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
        name = "nn_inference_node_palm_detector",
        parameters = [
        {
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": PathJoinSubstitution([model_path, "MediaPipeHandDetector.bin"])
        }],
        remappings = [
            ('/qrb_inference_input_tensor', '/palm_detector_input_tensor'),  # Remap input image topic
            ('/qrb_inference_output_tensor', '/palm_detector_output_tensor')  # Remap output depth topic
        ]
    )
    
    # Node for landmark detector
    nn_inference_node_landmark_detector = ComposableNode(
        package = "qrb_ros_nn_inference",
        plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
        name = "nn_inference_node_landmark_detector",
        parameters = [
        {
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": PathJoinSubstitution([model_path, "MediaPipeHandLandmarkDetector.bin"])
        }],
        remappings = [
            ('/qrb_inference_input_tensor', '/landmark_detector_input_tensor'),  # Remap input image topic
            ('/qrb_inference_output_tensor', '/landmark_detector_output_tensor')  # Remap output depth topic
        ]
    )

    # Create a ComposableNodeContainer to hold the inference nodes
    nn_inference_container = ComposableNodeContainer(
        name = "image_processing_container",
        package = "rclcpp_components",
        executable='component_container',
        namespace='',
        output = "screen",
        composable_node_descriptions = [nn_inference_node_palm_detector, nn_inference_node_landmark_detector]
    )

    # Node for image_publisher
    image_publisher_node = Node(
        package='image_publisher',  # Package name
        executable='image_publisher_node',  # Executable name
        name='image_publisher_node',  # Node name
        output='screen',  # Output logs to terminal
        parameters=[
            {'filename': image_path},  # Use the image_path argument
            {'rate': 10.0},  # Set the publishing rate to 10 Hz
        ],
    )

    # Node for sample_hand_detection
    hand_detector_node = Node(
        package='sample_hand_detection',  # Replace with the actual package name
        executable='qrb_ros_hand_detector',  # Replace with the actual executable name
        output='screen',  # Output logs to terminal
    )

    return LaunchDescription([
        image_path_arg,
        model_path_arg,
        nn_inference_container,
        image_publisher_node,
        hand_detector_node
    ])
