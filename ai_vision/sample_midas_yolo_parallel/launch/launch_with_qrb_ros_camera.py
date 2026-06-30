# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


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

    camera_info_path = PathJoinSubstitution([
        get_package_share_directory('qrb_ros_camera'),
        'config',
        'camera_info_imx577.yaml',
    ])

    camera_node = ComposableNode(
        package='qrb_ros_camera',
        namespace='',
        plugin='qrb_ros::camera::CameraNode',
        name='camera_node',
        parameters=[{
            'camera_id': 0,
            'stream_size': 1,
            'stream_name': ['stream1'],
            'stream1': {
                'height': 480,
                'width': 640,
                'fps': 30,
            },
            'camera_info_path': camera_info_path,
        }],
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

    container = ComposableNodeContainer(
        name='parallel_inference_camera_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[camera_node, midas_inference, yolo_inference],
        output='screen',
    )

    fusion_node = Node(
        package='sample_midas_yolo_parallel',
        executable='midas_yolo_parallel_node',
        name='midas_yolo_parallel_node',
        output='screen',
        parameters=[{
            'input_topic': '/cam0_stream1',
            'yolo_tensor_data_type': 4,
            'yolo_pack_uint16_input': True,
            'midas_input_size': [256, 256],
            'yolo_input_size': [640, 640],
            'score_thresh': 0.25,
            'iou_thresh': 0.45,
        }],
    )

    return LaunchDescription([
        midas_model_arg,
        yolo_model_arg,
        container,
        fusion_node,
    ])
