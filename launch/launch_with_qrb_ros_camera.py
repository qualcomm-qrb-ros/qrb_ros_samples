# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import LogInfo
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory


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

    # Node for qrb ros camera node
    camera_info_config_file_path =PathJoinSubstitution([
        get_package_share_directory('qrb_ros_camera'),
        'config', 'camera_info_imx577.yaml'])
    camera_node = ComposableNode(
        package='qrb_ros_camera',
        namespace=namespace,
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
            'camera_info_path': camera_info_config_file_path,
        }]
    )

    # Node fir qnn inference
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
            camera_node
        ]
    )

    return LaunchDescription(
        [
            model_path_arg,
            container, 
            depth_estimation_node
        ]
    )
