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
            DeclareLaunchArgument("rgb_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_raw"),
            DeclareLaunchArgument("mask_topic", default_value=""),
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
                        "rgb_topic": LaunchConfiguration("rgb_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "mask_topic": LaunchConfiguration("mask_topic"),
                    }
                ],
            ),
        ]
    )
