# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import LogInfo


def generate_launch_description():
    namespace = "sample_container"

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value="/opt/model/Depth-Anything-V2.bin",
        description='Path to the model file'
    )

    video_device_arg = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/video0',
        description='USB camera device path'
    )

    pixel_format_arg = DeclareLaunchArgument(
        'pixel_format',
        default_value='mjpeg2rgb',
        description='USB camera pixel format'
    )

    framerate_arg = DeclareLaunchArgument(
        'framerate',
        default_value='30.0',
        description='USB camera framerate'
    )

    image_width_arg = DeclareLaunchArgument(
        'image_width',
        default_value='640',
        description='USB camera image width'
    )

    image_height_arg = DeclareLaunchArgument(
        'image_height',
        default_value='480',
        description='USB camera image height'
    )

    model_path = LaunchConfiguration('model_path')
    video_device = LaunchConfiguration('video_device')
    pixel_format = LaunchConfiguration('pixel_format')
    framerate = LaunchConfiguration('framerate')
    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')

    LogInfo(msg=['MODEL_PATH: ', model_path])

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
        }]
    )

    depth_estimation_node = Node(
        package='sample_depth_estimation',
        executable='depth_estimation_node',
        name='depth_estimation_node',
        namespace=namespace,
    )

    nn_inference_node = ComposableNode(
        package="qrb_ros_nn_inference",
        namespace=namespace,
        plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
        name="nn_inference_node",
        parameters=[{
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": model_path,
            "log_level": "warn"
        }]
    )

    container = ComposableNodeContainer(
        name="container",
        namespace=namespace,
        package="rclcpp_components",
        executable='component_container',
        output="screen",
        composable_node_descriptions=[nn_inference_node],
        sigterm_timeout='3',
        sigkill_timeout='5'
    )

    return LaunchDescription(
        [
            model_path_arg,
            video_device_arg,
            pixel_format_arg,
            framerate_arg,
            image_width_arg,
            image_height_arg,
            usb_cam_node,
            container,
            depth_estimation_node
        ]
    )
