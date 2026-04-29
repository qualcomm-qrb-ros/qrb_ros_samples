# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("grasp_perception"), "config", "grasp_perception_node.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "file_input_config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("grasp_perception"), "config", "file_input_publisher.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "pose_onnx_path",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("grasp_perception"), "models", "ycb_models", "densefusion_ycb_posenet.onnx"]
                ),
            ),
            DeclareLaunchArgument(
                "refine_onnx_path",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("grasp_perception"), "models", "ycb_models", "densefusion_ycb_refiner.onnx"]
                ),
            ),
            DeclareLaunchArgument(
                "yolo_seg_onnx_path",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("grasp_perception"), "models", "yolo26n_seg_models", "yolo26n-seg.onnx"]
                ),
            ),
            # Directory mode: scene root that contains rgb/ and depth/ subfolders.
            DeclareLaunchArgument(
                "scene_dir",
                default_value="/home/data/qrb_ros_simulation_ws/DenseFusion-1/datasets/ycb-test-data/test~left_pbr/000045",
            ),
            DeclareLaunchArgument("mask_topic", default_value=""),
            Node(
                package="grasp_perception",
                executable="file_input_publisher",
                name="file_input_publisher",
                output="screen",
                parameters=[
                    LaunchConfiguration("file_input_config_file"),
                    {
                        "scene_dir": LaunchConfiguration("scene_dir"),
                        "rgb_topic": "/camera/color/image_raw",
                        "depth_topic": "/camera/depth/image_raw",
                    }
                ],
            ),
            Node(
                package="grasp_perception",
                executable="grasp_perception_node",
                name="grasp_perception_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("config_file"),
                    {
                        "pose_onnx_path": LaunchConfiguration("pose_onnx_path"),
                        "refine_onnx_path": LaunchConfiguration("refine_onnx_path"),
                        "yolo_seg_onnx_path": LaunchConfiguration("yolo_seg_onnx_path"),
                        "mask_topic": LaunchConfiguration("mask_topic"),
                        "rgb_topic": "/camera/color/image_raw",
                        "depth_topic": "/camera/depth/image_raw",
                        "cam_fx": 1066.778,
                        "cam_fy": 1067.487,
                        "cam_cx": 312.9869,
                        "cam_cy": 241.3109,
                        "depth_scale": 0.1,
                    }
                ],
            ),
        ]
    )
