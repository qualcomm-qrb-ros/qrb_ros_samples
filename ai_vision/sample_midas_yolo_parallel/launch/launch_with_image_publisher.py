# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def _read_target_model() -> str:
    for path in ('/proc/device-tree/model', '/sys/firmware/devicetree/base/model'):
        p = Path(path)
        if p.exists():
            return p.read_text(errors='ignore').lower().replace('\x00', ' ').strip()
    return ''


def _resolve_default_yolo_model_path() -> str:
    model_dir = Path('/opt/model')
    target_model = _read_target_model()

    iq8_candidate = model_dir / 'yolo11n-seg-w8a16-qcs8275-proxy.bin'
    iq9_candidate = model_dir / 'yolo11n-seg-w8a16-qcs9075.bin'

    if '8275' in target_model and iq8_candidate.exists():
        return str(iq8_candidate)
    if '9075' in target_model and iq9_candidate.exists():
        return str(iq9_candidate)

    for candidate in (
        iq8_candidate,
        iq9_candidate,
        model_dir / 'yolo11n-seg.bin',
        model_dir / 'yolov11_seg.bin',
    ):
        if candidate.exists():
            return str(candidate)

    discovered = sorted(model_dir.glob('*yolo*seg*.bin')) if model_dir.exists() else []
    if discovered:
        return str(discovered[0])

    return '/opt/model/yolo11n-seg.bin'


def generate_launch_description():
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value='/home/ubuntu/camera_raw_frame.jpg',
        description='Image used by image_publisher',
    )
    midas_model_arg = DeclareLaunchArgument(
        'midas_model_path',
        default_value='/opt/model/midas_256.bin',
        description='MiDaS QNN context binary path',
    )
    yolo_model_arg = DeclareLaunchArgument(
        'yolo_model_path',
        default_value=_resolve_default_yolo_model_path(),
        description='YOLO segmentation QNN context binary path',
    )
    pub_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='5.0',
        description='image_publisher rate',
    )

    image_pub = Node(
        package='image_publisher',
        executable='image_publisher_node',
        name='image_publisher_node',
        output='screen',
        parameters=[
            {'filename': LaunchConfiguration('image_path')},
            {'rate': LaunchConfiguration('publish_rate')},
        ],
    )

    midas_inference = ComposableNode(
        package='qrb_ros_nn_inference',
        plugin='qrb_ros::nn_inference::QrbRosInferenceNode',
        name='midas_inference_node',
        parameters=[{
            'backend_option': '/usr/lib/libQnnHtp.so',
            'model_path': LaunchConfiguration('midas_model_path'),
        }],
        remappings=[
            ('qrb_inference_input_tensor', 'midas_inference_input_tensor'),
            ('qrb_inference_output_tensor', 'midas_inference_output_tensor'),
        ],
    )

    yolo_inference = ComposableNode(
        package='qrb_ros_nn_inference',
        plugin='qrb_ros::nn_inference::QrbRosInferenceNode',
        name='yolo_seg_inference_node',
        parameters=[{
            'backend_option': '/usr/lib/libQnnHtp.so',
            'model_path': LaunchConfiguration('yolo_model_path'),
            'graph_name': 'yolov11_seg',
        }],
        remappings=[
            ('qrb_inference_input_tensor', 'yolo_seg_inference_input_tensor'),
            ('qrb_inference_output_tensor', 'yolo_seg_inference_output_tensor'),
        ],
    )

    inference_container = ComposableNodeContainer(
        name='parallel_inference_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[midas_inference, yolo_inference],
        output='screen',
    )

    fusion_node = Node(
        package='sample_midas_yolo_parallel',
        executable='midas_yolo_parallel_node',
        name='midas_yolo_parallel_node',
        output='screen',
        parameters=[{
            'input_topic': '/image_raw',
            'yolo_tensor_data_type': 4,
            'yolo_pack_uint16_input': True,
            'midas_input_size': [256, 256],
            'yolo_input_size': [640, 640],
            'score_thresh': 0.25,
            'iou_thresh': 0.45,
        }],
    )

    return LaunchDescription([
        image_path_arg,
        midas_model_arg,
        yolo_model_arg,
        pub_rate_arg,
        image_pub,
        inference_container,
        fusion_node,
    ])
