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

def generate_launch_description():
    # Declare the launch arguments for and model_path
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/opt/model/',
        description='Path to the model file'
    )

    camera_info_config_file_path = os.path.join(
        get_package_share_directory('qrb_ros_camera'),
        'config', 'camera_info_imx577.yaml'
    )

    model_path = LaunchConfiguration('model_path')

    # Node for face detector
    nn_inference_node_face_detector = ComposableNode(
        package = "qrb_ros_nn_inference",
        plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
        name = "nn_inference_node_face_detector",
        parameters = [
        {
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": PathJoinSubstitution([model_path, "MediaPipeFaceDetector.bin"])
        }],
        remappings = [
            ('/qrb_inference_input_tensor', '/face_detector_input_tensor'),  # Remap input image topic
            ('/qrb_inference_output_tensor', '/face_detector_output_tensor')  # Remap output depth topic
        ]
    )

    # Node for landmark detector
    nn_inference_node_face_landmark = ComposableNode(
        package = "qrb_ros_nn_inference",
        plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
        name = "nn_inference_node_face_landmark",
        parameters = [
        {
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": PathJoinSubstitution([model_path, "MediaPipeFaceLandmarkDetector.bin"])
        }],
        remappings = [
            ('/qrb_inference_input_tensor', '/face_landmark_input_tensor'),  # Remap input image topic
            ('/qrb_inference_output_tensor', '/face_landmark_output_tensor')  # Remap output depth topic
        ]
    )

    # Create a ComposableNodeContainer to hold the inference nodes
    nn_inference_container = ComposableNodeContainer(
        name = "image_processing_container",
        package = "rclcpp_components",
        executable='component_container',
        namespace='',
        output = "screen",
        composable_node_descriptions = [nn_inference_node_face_detector, nn_inference_node_face_landmark]
    )

    # Node for for qrb_ros_camera
    qrb_ros_camera_container = ComposableNodeContainer(
    name='my_container',
    namespace='',
    package='rclcpp_components',
    executable='component_container',
    composable_node_descriptions=[
        ComposableNode(
            package='qrb_ros_camera',
            plugin='qrb_ros::camera::CameraNode',
            name='camera_node',
            remappings=[('/camera_info', '/cam0_stream1_camera_info'),
                        ('/cam0_stream1', '/image_raw')],
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
            }]
            ),
    ],
    output='screen',
    )

    # Node for sample_face_detection
    face_detector_node = Node(
        package='sample_face_detection',  # Replace with the actual package name
        executable='qrb_ros_face_detector',  # Replace with the actual executable name
        output='screen',  # Output logs to terminal
    )

    return LaunchDescription([
        model_path_arg,
        qrb_ros_camera_container,
        nn_inference_container,
        face_detector_node
    ])