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
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import LogInfo
from launch.substitutions import TextSubstitution

def generate_launch_description():
    logger = get_logger('launch_with_qrb_ros_camera')

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value= "/opt/model/",
        description='Path to the model file'
    )
    model_path = LaunchConfiguration('model_path')
    
    config_dir = get_package_share_directory('qrb_ros_camera')

    camera_id_arg = DeclareLaunchArgument(
        'camera_id', default_value='5',
        description='Camera ID: inputId for QCarCam (e.g. 11/6)')

    width_arg  = DeclareLaunchArgument('width',  default_value='1280', description='Stream width in pixels')
    height_arg = DeclareLaunchArgument('height', default_value='720',  description='Stream height in pixels')
    fps_arg    = DeclareLaunchArgument('fps',    default_value='30',   description='Target frame rate')

    camera_info_path_arg = DeclareLaunchArgument(
        'camera_info_path',
        default_value=os.path.join(config_dir, 'config', 'camera_info_OX03F10_yuv.yaml'),
        description='Absolute path to camera intrinsic YAML file')

    dump_arg = DeclareLaunchArgument(
        'dump', default_value='False',
        description='Dump received frames to disk (debug)')

    camera_id        = LaunchConfiguration('camera_id')
    width            = LaunchConfiguration('width')
    height           = LaunchConfiguration('height')
    fps              = LaunchConfiguration('fps')
    camera_info_path = LaunchConfiguration('camera_info_path')
    dump             = LaunchConfiguration('dump')

    image_topic = [
        '/cam',
        camera_id,
        '_stream1'
    ]

    camera_info_topic = [
        '/cam',
        camera_id,
        '_stream1/camera_info'
    ]

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
                    'camera_id':        PythonExpression(["int('", camera_id, "')"]),
                    'stream_size':      1,
                    'stream_name':      ['stream1'],
                    'stream1.height':   PythonExpression(["int('", height, "')"]),
                    'stream1.width':    PythonExpression(["int('", width,  "')"]),
                    'stream1.fps':      PythonExpression(["int('", fps,    "')"]),
                    'camera_info_path': camera_info_path,
                }],
                remappings=[
                    (image_topic,'/image_raw')
                ],
            )
        ]
    )
    return LaunchDescription([
        model_path_arg,
        camera_id_arg,
        width_arg,
        height_arg,
        fps_arg,
        camera_info_path_arg,
        dump_arg,
        container,
        hr_pose_estimation_node,
        LogInfo(msg=image_topic)
    ])
