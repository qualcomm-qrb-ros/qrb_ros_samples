# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.logging import get_logger
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import LogInfo
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    logger = get_logger('launch_with_qrb_ros_camera')

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value= "/opt/model/",
        description='Path to the model file'
    )
    model_path = LaunchConfiguration('model_path')

    camera_info_config_file_path = os.path.join(
        get_package_share_directory('qrb_ros_camera'),
        'config', 'camera_info_OX03F10_yuv.yaml'
    )

    camera_info_path = camera_info_config_file_path

    hr_pose_estimation_node = Node(
        package='sample_hrnet_pose_estimation',
        executable='sample_hrnet_pose_estimation',
        output='screen',
    )

    container = ComposableNodeContainer(
        name="container",
        namespace='',
        package="rclcpp_components",
        executable="component_container",
        output='screen',
        composable_node_descriptions=[
            ComposableNode(
                package = "qrb_ros_nn_inference",
                plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
                name = "nn_inference_node",
                parameters=[
                    {
                        "backend_option": "/usr/lib/libQnnHtp.so",
                        "model_path": PathJoinSubstitution([model_path, "HRNetPose.bin"])
                    }
                ]
            ),
            ComposableNode(
                package='qrb_ros_camera',
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
                    'camera_info_path': camera_info_path,
                }],
                remappings=[
                    ('/cam0_stream1', '/image_raw'),
                    ('/cam0_stream1/camera_info', '/camera_info')
                ],
            )
        ]
    )

    return LaunchDescription([
        model_path_arg,
        container,
        hr_pose_estimation_node
    ])
