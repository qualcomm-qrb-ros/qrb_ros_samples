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
    # Declare the launch arguments for image_path and model_path
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value=os.path.join(get_package_share_directory('sample_face_detection'), 'face_image.jpg'),
        description='Path to the image file'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/opt/model/',
        description='Path to the model file'
    )

    image_path = LaunchConfiguration('image_path')
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
        namespace= '',
        name = "image_processing_container",
        package = "rclcpp_components",
        executable='component_container',
        output = "screen",
        composable_node_descriptions = [nn_inference_node_face_detector, nn_inference_node_face_landmark]
    )

    # Node for image_publisher
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

    # Node for sample_face_detection
    face_detector_node = Node(
        package='sample_face_detection',
        executable='qrb_ros_face_detector',
        output='screen',
    )

    return LaunchDescription([
        image_path_arg,
        model_path_arg,
        image_publisher_node,
        nn_inference_container,
        face_detector_node
    ])
