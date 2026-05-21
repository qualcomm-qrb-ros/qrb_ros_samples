# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Copyright (c) 2026, Qualcomm Innovation Center, Inc. and/or its affiliates.
# All rights reserved.

import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('sample_face_recognition')

    fd_model_path = PathJoinSubstitution([
        pkg_share, 'model', 'face_detection_yunet_2021dec_static.onnx'
    ])
    fr_model_path = PathJoinSubstitution([
        pkg_share, 'model', 'face_recognition_sface_2021dec_static.onnx'
    ])
    image_data_path = PathJoinSubstitution([
        pkg_share, 'resource', 'data'
    ])

    composable_nodes = [
        ComposableNode(
            package='sample_face_recognition',
            plugin='sample_face_recognition::RecognitionNode',
            name='',
            parameters=[{
                'overlay': 1,
                'similar_threshold': 0.363,
                'fd_model': fd_model_path,
                'fr_model': fr_model_path,
                'image_data_path': image_data_path,
                'fps_max': 30,
            }]
        ),
    ]

    container = ComposableNodeContainer(
        name="sample_face_recognition_node",
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen'
    )

    return launch.LaunchDescription([container])