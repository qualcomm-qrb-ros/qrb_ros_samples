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
    sample_namespace = "sample_hand_detection"
    # Declare the launch arguments for and model_path
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/opt/model/',
        description='Path to the model file'
    )
    
    # Use LaunchConfiguration to get the values of the arguments
    model_path = LaunchConfiguration('model_path')



    # Node for for qrb_ros_camera
    camera_info_config_file_path = os.path.join(
        get_package_share_directory('qrb_ros_camera'),
        'config', 'camera_info_imx577.yaml'
    )

    camera_node_params = {
        'camera_id': 0,
        'stream_size': 1,
        'stream_name': ['stream1'],
        'stream1': {
            'height': 480,
            'width': 640,
            'fps': 5,
        },
        'camera_info_path': camera_info_config_file_path,
    }

    qrb_ros_camera_container = ComposableNodeContainer(
    name='camera_container',
    namespace='',
    package='rclcpp_components',
    executable='component_container',
    composable_node_descriptions=[
        ComposableNode(
            package='qrb_ros_camera',
            plugin='qrb_ros::camera::CameraNode',
            name='camera_node',
            parameters=[camera_node_params]
            ),
    ],
    output='screen',
    )

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

    # Node for sample_hand_detection
    hand_detector_node = Node(
        package='sample_hand_detection',  # Replace with the actual package name
        executable='qrb_ros_hand_detector',  # Replace with the actual executable name
        output='screen',  # Output logs to terminal
        remappings=[
            ('/image_raw', '/cam0_stream1'),  # Remap the image topic
        ],
        parameters=[
            {'model_path': model_path}  # Pass the model path as a parameter
        ]
    )

    return LaunchDescription([
        model_path_arg,
        qrb_ros_camera_container,
        nn_inference_container,
        hand_detector_node
    ])
