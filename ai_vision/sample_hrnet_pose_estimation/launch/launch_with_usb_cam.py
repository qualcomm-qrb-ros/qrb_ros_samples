# Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value="/opt/model/",
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

    hr_pose_estimation_node = Node(
        package='sample_hrnet_pose_estimation',
        executable='sample_hrnet_pose_estimation',
        output='screen',
    )

    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam_node',
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

    nn_inference_container = ComposableNodeContainer(
        name="container",
        namespace='',
        package="rclcpp_components",
        executable="component_container",
        output='screen',
        composable_node_descriptions=[
            ComposableNode(
                package="qrb_ros_nn_inference",
                plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
                name="nn_inference_node",
                parameters=[{
                    "backend_option": "/usr/lib/libQnnHtp.so",
                    "model_path": PathJoinSubstitution([model_path, "HRNetPose.bin"])
                }]
            )
        ]
    )

    return LaunchDescription([
        model_path_arg,
        video_device_arg,
        pixel_format_arg,
        framerate_arg,
        image_width_arg,
        image_height_arg,
        usb_cam_node,
        nn_inference_container,
        hr_pose_estimation_node
    ])
