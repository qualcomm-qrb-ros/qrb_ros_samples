# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            'model_path',
            default_value="/opt/model/ResNet101_w8a8.bin",
            description='Path to the model file'
        ),
        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video0',
            description='USB camera device path'
        ),
        DeclareLaunchArgument(
            'pixel_format',
            default_value='mjpeg2rgb',
            description='USB camera pixel format'
        ),
        DeclareLaunchArgument(
            'framerate',
            default_value='30.0',
            description='USB camera framerate'
        ),
        DeclareLaunchArgument(
            'image_width',
            default_value='640',
            description='USB camera image width'
        ),
        DeclareLaunchArgument(
            'image_height',
            default_value='480',
            description='USB camera image height'
        ),
    ]

    model_path = LaunchConfiguration('model_path')
    video_device = LaunchConfiguration('video_device')
    pixel_format = LaunchConfiguration('pixel_format')
    framerate = LaunchConfiguration('framerate')
    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')

    namespace = ""

    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam_node',
        namespace=namespace,
        output='screen',
        parameters=[{
            'video_device': video_device,
            'pixel_format': pixel_format,
            'framerate': framerate,
            'image_width': image_width,
            'image_height': image_height,
            'io_method': 'mmap',
            'frame_id': 'camera',
            'brightness': -1,
            'contrast': -1,
            'saturation': -1,
            'sharpness': -1,
            'gain': -1,
            'focus': -1,
        }]
    )

    preprocess_node = Node(
        package='sample_resnet101',
        executable='qrb_ros_resnet101',
        namespace=namespace,
        output='screen',
    )

    nn_inference_node = ComposableNode(
        package="qrb_ros_nn_inference",
        namespace=namespace,
        plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
        name="nn_inference_node",
        parameters=[{
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": model_path
        }]
    )

    postprocess_node = Node(
        package='sample_resnet101',
        executable='qrb_ros_resnet101_posprocess',
        namespace=namespace,
        output='screen',
    )

    container = ComposableNodeContainer(
        name="container",
        namespace=namespace,
        package="rclcpp_components",
        executable='component_container',
        output="screen",
        composable_node_descriptions=[nn_inference_node]
    )

    return launch.LaunchDescription(declared_args + [usb_cam_node, preprocess_node, container, postprocess_node])
