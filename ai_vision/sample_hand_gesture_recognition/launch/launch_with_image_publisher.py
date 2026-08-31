# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare the launch arguments for image_path and model_path
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value=os.path.join(
            get_package_share_directory('sample_hand_gesture_recognition'),
            'input_image.jpg'),
        description='Path to the image file'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/opt/model/',
        description='Directory containing the palm/landmark models and gesture weights'
    )

    image_path = LaunchConfiguration('image_path')
    model_path = LaunchConfiguration('model_path')

    # Stage 1: palm detector on the NPU
    nn_inference_node_palm_detector = ComposableNode(
        package="qrb_ros_nn_inference",
        plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
        name="nn_inference_node_palm_detector",
        parameters=[{
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": PathJoinSubstitution([model_path, "palm_detector.bin"])
        }],
        remappings=[
            ('/qrb_inference_input_tensor', '/palm_detector_input_tensor'),
            ('/qrb_inference_output_tensor', '/palm_detector_output_tensor')
        ]
    )

    # Stage 2: hand landmark detector on the NPU
    nn_inference_node_landmark_detector = ComposableNode(
        package="qrb_ros_nn_inference",
        plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
        name="nn_inference_node_landmark_detector",
        parameters=[{
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": PathJoinSubstitution([model_path, "hand_landmark_detector.bin"])
        }],
        remappings=[
            ('/qrb_inference_input_tensor', '/landmark_detector_input_tensor'),
            ('/qrb_inference_output_tensor', '/landmark_detector_output_tensor')
        ]
    )

    # Stage 3 (gesture classification) runs in-node as a small FP32 model, so it
    # does not need an inference container here.
    nn_inference_container = ComposableNodeContainer(
        name="gesture_inference_container",
        package="rclcpp_components",
        executable='component_container',
        namespace='',
        output="screen",
        composable_node_descriptions=[
            nn_inference_node_palm_detector,
            nn_inference_node_landmark_detector,
        ]
    )

    # Publish a still image to /image_raw on a loop
    image_publisher_node = Node(
        package='image_publisher',
        executable='image_publisher_node',
        name='image_publisher_node',
        output='screen',
        parameters=[
            {'filename': image_path},
            {'rate': 5.0},
        ],
    )

    # The gesture recognition node: pre/post-processing, staging, and stage-3 FP32 classify
    gesture_recognition_node = Node(
        package='sample_hand_gesture_recognition',
        executable='qrb_ros_gesture_recognition',
        output='screen',
        parameters=[
            {'model_path': model_path}
        ]
    )

    return LaunchDescription([
        image_path_arg,
        model_path_arg,
        nn_inference_container,
        image_publisher_node,
        gesture_recognition_node
    ])
