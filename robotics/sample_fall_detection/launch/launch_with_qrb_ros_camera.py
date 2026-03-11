# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # -------- QRB ROS Camera (IMX577 example) --------
    camera_info_config_file_path = PathJoinSubstitution([
        get_package_share_directory('qrb_ros_camera'),
        'config', 'camera_info_imx577.yaml'
    ])

    camera_node = ComposableNode(
        package='qrb_ros_camera',
        namespace="",
        plugin='qrb_ros::camera::CameraNode',
        name='camera_node',
        parameters=[{
            'camera_id': 0,
            'stream_size': 1,
            'stream_name': ["stream1"],
            'stream1': {
                'height': 480,
                'width': 640,
                'fps': 30,
            },
            'camera_info_path': camera_info_config_file_path,
            'dump': False,
            'dump_camera_info_': False,
        }]
    )

    camera_container = ComposableNodeContainer(
        name='camera_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[camera_node],
        output='screen',
    )

    # -------- Fall detection node (Python) --------
    fall_node = Node(
        package='sample_fall_detection',
        executable='fall_state_gate_node',
        name='fall_state_gate_node',
        output='screen',
        parameters=[{
            'image_topic': '/cam0_stream1',
            'state_service': '/fall_detection/state_control',
            'timer_period_sec': 0.03,

            'save_dir': '/tmp/fall_images',
            'save_on_fall_only': True,
            'save_ext': '.jpg',
            'save_once_per_frame': True,
            'fall_save_cooldown_sec': 0.2,
            'save_visualized': True,

            'republish_image': True,
            'republish_topic': '/fall_detection/image',
            'republish_encoding': 'bgr8',
            'default_frame_id': 'camera',

            'fall_pub_topic': '/fall_detected',
            'publish_false': False,

            'enable_inference': True,
            'model_path': '/opt/model/fall-detection-with-rb8-linux-aarch64-qnn-v1.eim',
            'ei_debug': False,

            'fall_label': 'fallen',
            'fall_threshold': 0.55,
        }]
    )

    return LaunchDescription([
        camera_container,
        fall_node,
    ])
