# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import LogInfo
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch.actions import TimerAction


def generate_launch_description():
    namespace = "sample_container"
    # Declare the launch arguments for image_path and model_path
    model_path_arg = DeclareLaunchArgument(
       'model_path',
        default_value="/opt/model/Depth-Anything-V2.bin",
        description='Path to the model file'
    )
    model_path= LaunchConfiguration('model_path')
    LogInfo(msg=['MODEL_PATH: ', model_path])

    # Node for depth_estimation
    depth_estimation_node = Node(
        package='sample_depth_estimation',
        executable='depth_estimation_node', 
        name='depth_estimation_node', 
        namespace=namespace, 
    )
    
    delayed_depth_estimation_node = TimerAction(
        period=3.0,
        actions=[depth_estimation_node],
    )
    
    # Node for qrb ros camera node
    camera_id_arg = DeclareLaunchArgument(
        'camera_id', default_value='5',
        description='Camera ID: inputId for QCarCam (e.g. 11/6)'
    )

    width_arg  = DeclareLaunchArgument('width',  default_value='1280', description='Stream width in pixels')
    height_arg = DeclareLaunchArgument('height', default_value='720',  description='Stream height in pixels')
    fps_arg    = DeclareLaunchArgument('fps',    default_value='30',   description='Target frame rate')

    camera_info_path_arg = DeclareLaunchArgument(
        'camera_info_path',
        default_value=PathJoinSubstitution([get_package_share_directory('qrb_ros_camera'), 
                                            'config', 'camera_info_OX03F10_yuv.yaml']),
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

    camera_node = ComposableNode(
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

    # Node for qnn inference
    nn_inference_node = ComposableNode(
        package = "qrb_ros_nn_inference",
        namespace=namespace,
        plugin = "qrb_ros::nn_inference::QrbRosInferenceNode",
        name = "nn_inference_node",
        parameters = [
        {
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": model_path
        }]
    )

    container = ComposableNodeContainer(
        name = "container",
        namespace=namespace,
        package = "rclcpp_components",
        executable='component_container',
        output = "screen",
        composable_node_descriptions = [
            nn_inference_node, 
            camera_node,
        ]
    )

    return LaunchDescription(
        [
            model_path_arg,
            camera_id_arg,
            width_arg,
            height_arg,
            fps_arg,
            camera_info_path_arg,
            dump_arg,
            container,
            delayed_depth_estimation_node
        ]
    )
