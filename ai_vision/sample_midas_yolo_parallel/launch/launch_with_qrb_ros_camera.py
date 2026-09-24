# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


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
    score_thresh_arg = DeclareLaunchArgument(
        'score_thresh',
        default_value='0.25',
        description='YOLO detection score threshold',
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

    fusion_cpp = ComposableNode(
        package='sample_midas_yolo_parallel',
        plugin='sample_midas_yolo_parallel::MidasYoloFusionNode',
        name='midas_yolo_fusion_node',
        parameters=[{
            'input_topic': '/cam0_stream1',
            'yolo_tensor_data_type': 2,
            'yolo_pack_uint16_input': False,
            'midas_input_size': [256, 256],
            'yolo_input_size': [640, 640],
            'score_thresh': LaunchConfiguration('score_thresh'),
            'iou_thresh': 0.45,
        }],
    )

    container = ComposableNodeContainer(
        name='parallel_inference_camera_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[camera_node, shared_inference, fusion_cpp],
        output='screen',
    )

    return LaunchDescription([
        combined_model_arg,
        midas_graph_name_arg,
        yolo_graph_name_arg,
        score_thresh_arg,
        container,
    ])
