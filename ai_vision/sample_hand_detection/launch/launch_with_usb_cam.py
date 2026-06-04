# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/opt/model/',
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

    nn_inference_node_palm_detector = ComposableNode(
        package="qrb_ros_nn_inference",
        plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
        name="nn_inference_node_palm_detector",
        parameters=[{
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": PathJoinSubstitution([model_path, "MediaPipeHandDetector.bin"])
        }],
        remappings=[
            ('/qrb_inference_input_tensor', '/palm_detector_input_tensor'),
            ('/qrb_inference_output_tensor', '/palm_detector_output_tensor')
        ]
    )

    nn_inference_node_landmark_detector = ComposableNode(
        package="qrb_ros_nn_inference",
        plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
        name="nn_inference_node_landmark_detector",
        parameters=[{
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": PathJoinSubstitution([model_path, "MediaPipeHandLandmarkDetector.bin"])
        }],
        remappings=[
            ('/qrb_inference_input_tensor', '/landmark_detector_input_tensor'),
            ('/qrb_inference_output_tensor', '/landmark_detector_output_tensor')
        ]
    )

    nn_inference_container = ComposableNodeContainer(
        name="image_processing_container",
        package="rclcpp_components",
        executable='component_container',
        namespace='',
        output="screen",
        composable_node_descriptions=[nn_inference_node_palm_detector, nn_inference_node_landmark_detector]
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

    hand_detector_node = Node(
        package='sample_hand_detection',
        executable='qrb_ros_hand_detector',
        output='screen',
        parameters=[
            {'model_path': model_path}
        ]
    )

    return LaunchDescription([
        model_path_arg,
        video_device_arg,
        pixel_format_arg,
        framerate_arg,
        image_width_arg,
        image_height_arg,
        nn_inference_container,
        usb_cam_node,
        hand_detector_node
    ])
