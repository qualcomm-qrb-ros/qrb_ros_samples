# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    combined_model_arg = DeclareLaunchArgument(
        'combined_model_path',
        default_value='/opt/model/midas_yolo_combined_int8_split.bin',
        description='Combined multi-graph QNN context binary containing MiDaS and YOLO graphs',
    )
    midas_graph_name_arg = DeclareLaunchArgument(
        'midas_graph_name',
        default_value='midas',
        description='Graph name for MiDaS depth estimation within the combined context binary',
    )
    yolo_graph_name_arg = DeclareLaunchArgument(
        'yolo_graph_name',
        default_value='yolov11_seg',
        description='Graph name for YOLO segmentation within the combined context binary',
    )

    video_device_arg = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/video0',
        description='USB camera device path',
    )
    pixel_format_arg = DeclareLaunchArgument(
        'pixel_format',
        default_value='mjpeg2rgb',
        description='USB camera pixel format',
    )
    framerate_arg = DeclareLaunchArgument(
        'framerate',
        default_value='30.0',
        description='USB camera framerate',
    )
    image_width_arg = DeclareLaunchArgument(
        'image_width',
        default_value='640',
        description='USB camera image width',
    )
    image_height_arg = DeclareLaunchArgument(
        'image_height',
        default_value='480',
        description='USB camera image height',
    )
    brightness_arg = DeclareLaunchArgument(
        'brightness',
        default_value='-1',
        description='USB camera brightness (-1 to disable)',
    )
    contrast_arg = DeclareLaunchArgument(
        'contrast',
        default_value='-1',
        description='USB camera contrast (-1 to disable)',
    )
    saturation_arg = DeclareLaunchArgument(
        'saturation',
        default_value='-1',
        description='USB camera saturation (-1 to disable)',
    )
    sharpness_arg = DeclareLaunchArgument(
        'sharpness',
        default_value='-1',
        description='USB camera sharpness (-1 to disable)',
    )
    gain_arg = DeclareLaunchArgument(
        'gain',
        default_value='-1',
        description='USB camera gain (-1 to disable)',
    )
    focus_arg = DeclareLaunchArgument(
        'focus',
        default_value='-1',
        description='USB camera focus (-1 to disable)',
    )

    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam_node',
        output='screen',
        parameters=[{
            'video_device': LaunchConfiguration('video_device'),
            'pixel_format': LaunchConfiguration('pixel_format'),
            'framerate': LaunchConfiguration('framerate'),
            'image_width': LaunchConfiguration('image_width'),
            'image_height': LaunchConfiguration('image_height'),
            'io_method': 'mmap',
            'frame_id': 'camera',
            'brightness': LaunchConfiguration('brightness'),
            'contrast': LaunchConfiguration('contrast'),
            'saturation': LaunchConfiguration('saturation'),
            'sharpness': LaunchConfiguration('sharpness'),
            'gain': LaunchConfiguration('gain'),
            'focus': LaunchConfiguration('focus'),
        }],
    )

    shared_inference = ComposableNode(
        package='qrb_ros_nn_inference',
        plugin='qrb_ros::nn_inference::QrbRosSharedInferenceNode',
        name='shared_inference_node',
        parameters=[{
            'backend_option': '/usr/lib/libQnnHtp.so',
            'model_path': LaunchConfiguration('combined_model_path'),
            'graph_name_0': LaunchConfiguration('midas_graph_name'),
            'graph_name_1': LaunchConfiguration('yolo_graph_name'),
        }],
        remappings=[
            # graph 0 = MiDaS
            ('qrb_inference_input_tensor_0',  'midas_inference_input_tensor'),
            ('qrb_inference_output_tensor_0', 'midas_inference_output_tensor'),
            # graph 1 = YOLO
            ('qrb_inference_input_tensor_1',  'yolo_seg_inference_input_tensor'),
            ('qrb_inference_output_tensor_1', 'yolo_seg_inference_output_tensor'),
        ],
    )

    # C++ fusion node loaded into the SAME container as the inference node.
    # This enables intra-process zero-copy for all tensor messages — no
    # serialization overhead between inference outputs and fusion inputs.
    fusion_cpp = ComposableNode(
        package='sample_midas_yolo_parallel_cpp',
        plugin='sample_midas_yolo_parallel_cpp::MidasYoloFusionNode',
        name='midas_yolo_fusion_node',
        parameters=[{
            'input_topic': '/image_raw',
            'yolo_tensor_data_type': 2,
            'yolo_pack_uint16_input': False,
            'midas_input_size': [256, 256],
            'yolo_input_size': [640, 640],
            'score_thresh': 0.25,
            'iou_thresh': 0.45,
        }],
    )

    # usb_cam publishes /image_raw on a standalone node; inference + fusion
    # share one container for intra-process zero-copy tensor delivery.
    inference_container = ComposableNodeContainer(
        name='parallel_inference_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[shared_inference, fusion_cpp],
        output='screen',
    )

    return LaunchDescription([
        combined_model_arg,
        midas_graph_name_arg,
        yolo_graph_name_arg,
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
        inference_container,
    ])
