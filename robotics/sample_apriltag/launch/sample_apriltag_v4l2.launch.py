# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    video_device = LaunchConfiguration('video_device')
    image_size = LaunchConfiguration('image_size')
    apriltag_conf = LaunchConfiguration('apriltag_conf')
    camera_info_path = LaunchConfiguration('camera_info_path')

    video_device_launch_arg = DeclareLaunchArgument(
        'video_device', default_value='/dev/video6'
    )
    image_size_launch_arg = DeclareLaunchArgument(
        'image_size', default_value='[640, 480]'
    )
    camera_info_path_launch_arg = DeclareLaunchArgument(
        'camera_info_path', default_value="file://" + os.path.join(
            get_package_share_directory("sample_apriltag"),
            "config",
            "realsense_depth_435i_640x480.yaml"
        )
    )
    apriltag_conf_launch_arg = DeclareLaunchArgument(
      'apriltag_conf', default_value=os.path.join(
            get_package_share_directory("sample_apriltag"),
            "config",
            "tags_41h12.yaml",
        )
    )

    v4l2_camera = ComposableNode(
        name='camera',
        package='v4l2_camera',
        plugin='v4l2_camera::V4L2Camera',
        namespace='apriltag',
        parameters=[{
          "video_device": video_device,
          "image_size": image_size,
          "camera_info_url": camera_info_path,
        }],
        extra_arguments=[{
          "use_intra_process_comms": True
        }],
    )

    rectify = ComposableNode(
        name='rectify',
        package='image_proc',
        plugin='image_proc::RectifyNode',
        namespace='apriltag',
        extra_arguments=[{
          "use_intra_process_comms": True
        }],
        remappings=[("image", "image_raw")],
    )

    apriltag_ros = ComposableNode(
        name='apriltag_ros',
        package='apriltag_ros',
        plugin='AprilTagNode',
        namespace='apriltag',
        parameters=[apriltag_conf],
        extra_arguments=[{
          "use_intra_process_comms": True
        }],
    )

    container = ComposableNodeContainer(
            name='qrb_ros_apriltag_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                v4l2_camera,
                rectify,
                apriltag_ros,
            ],
            output='screen'
    )

    return launch.LaunchDescription([
        video_device_launch_arg,
        image_size_launch_arg,
        camera_info_path_launch_arg,
        apriltag_conf_launch_arg,
        container,
    ])