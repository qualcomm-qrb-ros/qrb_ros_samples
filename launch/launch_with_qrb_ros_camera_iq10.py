# Copyright (c) 2025 Qualcomm Innovation Center, Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    namespace = ""

    #
    # Launch Arguments
    #
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="/opt/model/ResNet101_w8a8.bin",
        description="Path to the model file",
    )

    camera_id_arg = DeclareLaunchArgument(
        "camera_id",
        default_value="5",
        description="Camera ID for QCarCam",
    )

    model_path = LaunchConfiguration("model_path")
    camera_id = LaunchConfiguration("camera_id")

    #
    # Camera Info Config
    #
    camera_info_config_file_path = PathJoinSubstitution(
        [
            get_package_share_directory("qrb_ros_camera"),
            "config",
            "camera_info_imx577.yaml",
        ]
    )

    #
    # Topics
    #
    image_topic = [
        "/cam",
        camera_id,
        "_stream1",
    ]

    camera_info_topic = [
        "/cam",
        camera_id,
        "_stream1/camera_info",
    ]

    #
    # QRB Camera Node
    #
    camera_node = ComposableNode(
        package="qrb_ros_camera",
        plugin="qrb_ros::camera::CameraNode",
        namespace=namespace,
        name="camera_node",
        parameters=[
            {
                "camera_id": camera_id,
                "stream_size": 1,
                "stream_name": ["stream1"],
                "stream1": {
                    "height": 720,
                    "width": 1280,
                    "fps": 30,
                },
                "camera_info_path": camera_info_config_file_path,
                "dump": False,
                "dump_camera_info": False,
            }
        ],
    )

    #
    # Preprocess Node
    #
    preprocess_node = Node(
        package="sample_resnet101",
        executable="qrb_ros_resnet101",
        namespace=namespace,
        output="screen",
        remappings=[
            ("image_raw", image_topic),
        ],
    )

    #
    # QNN Inference Node
    #
    nn_inference_node = ComposableNode(
        package="qrb_ros_nn_inference",
        plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
        namespace=namespace,
        name="nn_inference_node",
        parameters=[
            {
                "backend_option": "/usr/lib/libQnnHtp.so",
                "model_path": model_path,
            }
        ],
    )

    #
    # Postprocess Node
    #
    postprocess_node = Node(
        package="sample_resnet101",
        executable="qrb_ros_resnet101_posprocess",
        namespace=namespace,
        output="screen",
    )

    #
    # Component Container
    #
    container = ComposableNodeContainer(
        name="container",
        namespace=namespace,
        package="rclcpp_components",
        executable="component_container",
        output="screen",
        composable_node_descriptions=[
            camera_node,
            nn_inference_node,
        ],
    )

    #
    # Launch Description
    #
    return LaunchDescription(
        [
            model_path_arg,
            camera_id_arg,
            preprocess_node,
            container,
            postprocess_node,
        ]
    )
