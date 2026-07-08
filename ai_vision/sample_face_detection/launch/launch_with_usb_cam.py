# Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
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

    brightness_arg = DeclareLaunchArgument(
        'brightness',
        default_value='-1',
        description='USB camera brightness (-1 to disable)'
    )

    contrast_arg = DeclareLaunchArgument(
        'contrast',
        default_value='-1',
        description='USB camera contrast (-1 to disable)'
    )

    saturation_arg = DeclareLaunchArgument(
        'saturation',
        default_value='-1',
        description='USB camera saturation (-1 to disable)'
    )

    sharpness_arg = DeclareLaunchArgument(
        'sharpness',
        default_value='-1',
        description='USB camera sharpness (-1 to disable)'
    )

    gain_arg = DeclareLaunchArgument(
        'gain',
        default_value='-1',
        description='USB camera gain (-1 to disable)'
    )

    focus_arg = DeclareLaunchArgument(
        'focus',
        default_value='-1',
        description='USB camera focus (-1 to disable)'
    )

    model_path = LaunchConfiguration('model_path')
    video_device = LaunchConfiguration('video_device')
    pixel_format = LaunchConfiguration('pixel_format')
    framerate = LaunchConfiguration('framerate')
    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')
    brightness = LaunchConfiguration('brightness')
    contrast = LaunchConfiguration('contrast')
    saturation = LaunchConfiguration('saturation')
    sharpness = LaunchConfiguration('sharpness')
    gain = LaunchConfiguration('gain')
    focus = LaunchConfiguration('focus')

    nn_inference_node_face_detector = ComposableNode(
        package="qrb_ros_nn_inference",
        plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
        name="nn_inference_node_face_detector",
        parameters=[{
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": PathJoinSubstitution([model_path, "MediaPipeFaceDetector.bin"])
        }],
        remappings=[
            ('/qrb_inference_input_tensor', '/face_detector_input_tensor'),
            ('/qrb_inference_output_tensor', '/face_detector_output_tensor')
        ]
    )

    nn_inference_node_face_landmark = ComposableNode(
        package="qrb_ros_nn_inference",
        plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
        name="nn_inference_node_face_landmark",
        parameters=[{
            "backend_option": "/usr/lib/libQnnHtp.so",
            "model_path": PathJoinSubstitution([model_path, "MediaPipeFaceLandmarkDetector.bin"])
        }],
        remappings=[
            ('/qrb_inference_input_tensor', '/face_landmark_input_tensor'),
            ('/qrb_inference_output_tensor', '/face_landmark_output_tensor')
        ]
    )

    nn_inference_container = ComposableNodeContainer(
        namespace='',
        name="image_processing_container",
        package="rclcpp_components",
        executable='component_container',
        output="screen",
        composable_node_descriptions=[nn_inference_node_face_detector, nn_inference_node_face_landmark]
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
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'sharpness': sharpness,
            'gain': gain,
            'focus': focus,
        }]
    )

    face_detector_node = Node(
        package='sample_face_detection',
        executable='qrb_ros_face_detector',
        output='screen',
    )

    return LaunchDescription([
        model_path_arg,
        video_device_arg,
        pixel_format_arg,
        framerate_arg,
        image_width_arg,
        image_height_arg,
        brightness_arg,
        contrast_arg,
        saturation_arg,
        sharpness_arg,
        gain_arg,
        focus_arg,
        usb_cam_node,
        nn_inference_container,
        face_detector_node
    ])
