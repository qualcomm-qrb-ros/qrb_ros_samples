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
    camera_id = LaunchConfiguration('camera_id')
    camera_id_launch_arg = DeclareLaunchArgument(
        'camera_id', default_value='0'
    )

    apriltag_conf = LaunchConfiguration('apriltag_conf')
    apriltag_conf_launch_arg = DeclareLaunchArgument(
      'apriltag_conf', default_value=os.path.join(
            get_package_share_directory("sample_apriltag"),
            "config",
            "tags_41h12.yaml",
        )
    )

    camera_info_file_path = os.path.join(
        get_package_share_directory('qrb_ros_camera'),
        'config', 'camera_info_imx577.yaml'
    )
    camera = ComposableNode(
        name='camera',
        package='qrb_ros_camera',
        plugin='qrb_ros::camera::CameraNode',
        namespace='apriltag',
        parameters=[{
            'camera_id': camera_id,
            'stream_size': 1,
            'stream_name': ["stream1"],
            'stream1':{
                'height':720,
                'width':1280,
                'fps':30,
            },
            'camera_info_path': camera_info_file_path,
        }],
        extra_arguments=[{'use_intra_process_comms': True}],
        remappings=[
          ("cam0_stream1", "image_nv12"),
          ("cam0_stream1_camera_info", "camera_info"),
        ],
    )

    color_convert = ComposableNode(
        name='color_convert',
        package='qrb_ros_colorspace_convert',
        plugin='qrb_ros::colorspace_convert::ColorspaceConvertNode',
        namespace='apriltag',
        parameters=[{
            'conversion_type': 'nv12_to_rgb8',
            'latency_fps_test': False,
        }],
        extra_arguments=[{'use_intra_process_comms': True}],
        remappings=[
          ("image_raw", "image_nv12"),
          ("image", "image_rgb8"),
        ],
    )

    rectify = ComposableNode(
        name='rectify',
        package='image_proc',
        plugin='image_proc::RectifyNode',
        namespace='apriltag',
        extra_arguments=[{
          "use_intra_process_comms": True
        }],
        remappings=[
            ("image", "image_rgb8"),
        ],
    )

    apriltag_ros = ComposableNode(
        name='apriltag',
        package='apriltag_ros',
        plugin='AprilTagNode',
        namespace='apriltag',
        parameters=[apriltag_conf],
        extra_arguments=[{
          "use_intra_process_comms": True
        }],
    )

    container = ComposableNodeContainer(
        name='apriltag_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            camera,
            color_convert,
            rectify,
            apriltag_ros,
        ],
        output='screen'
    )

    return launch.LaunchDescription([
        camera_id_launch_arg,
        apriltag_conf_launch_arg,
        container,
    ])
