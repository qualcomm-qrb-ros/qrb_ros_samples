# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # Declare the launch argument for model_path
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/opt/model/',
        description='Directory containing the palm/landmark models and gesture weights'
    )

    model_path = LaunchConfiguration('model_path')

    # qrb_ros_camera: capture frames and publish to /cam0_stream1
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
            'fps': 30,
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

    nn_inference_container = ComposableNodeContainer(
        name="image_processing_container",
        package="rclcpp_components",
        executable='component_container',
        namespace='',
        output="screen",
        composable_node_descriptions=[
            nn_inference_node_palm_detector,
            nn_inference_node_landmark_detector,
        ]
    )

    # The gesture recognition node (stage-3 FP32 classify runs in-node)
    gesture_recognition_node = Node(
        package='sample_hand_gesture_recognition',
        executable='qrb_ros_gesture_recognition',
        output='screen',
        remappings=[
            ('/image_raw', '/cam0_stream1'),
        ],
        parameters=[
            {'model_path': model_path}
        ]
    )

    return LaunchDescription([
        model_path_arg,
        qrb_ros_camera_container,
        nn_inference_container,
        gesture_recognition_node
    ])
