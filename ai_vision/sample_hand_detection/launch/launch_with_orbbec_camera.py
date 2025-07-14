# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Declare the launch arguments for and model_path
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/opt/model/',
        description='Path to the model file'
    )
    
    # Use LaunchConfiguration to get the values of the arguments
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

    #orbbec camera
    orbbec_camera_launch_path = os.path.join(
        get_package_share_directory('orbbec_camera'),
        'launch',
        'gemini_330_series.launch.py'
    )
    orbbec_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(orbbec_camera_launch_path),
        launch_arguments={
            'color_width': '640',
            'color_height': '480',
            'color_fps': '30',
            'color_qos': 'default'
        }.items()
    )

    # Node for sample_hand_detection
    hand_detector_node = Node(
        package='sample_hand_detection',  # Replace with the actual package name
        executable='qrb_ros_hand_detector',  # Replace with the actual executable name
        output='screen',  # Output logs to terminal
        remappings=[
            ('/image_raw', '/camera/color/image_raw'),  # Remap the image topic
        ],
    )

    return LaunchDescription([
        model_path_arg,
        nn_inference_container,
        orbbec_camera_launch,
        hand_detector_node
    ])
