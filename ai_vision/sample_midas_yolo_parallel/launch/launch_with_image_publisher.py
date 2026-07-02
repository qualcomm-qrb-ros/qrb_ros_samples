# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value='/home/ubuntu/camera_raw_frame.jpg',
        description='Image used by image_publisher',
    )
    combined_model_arg = DeclareLaunchArgument(
        'combined_model_path',
        default_value='/opt/model/midas_yolo_combined.bin',
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
    pub_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='5.0',
        description='image_publisher rate',
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

    # FastImagePublisherNode: pre-loads the image once, publishes at full rate
    # without per-frame JPEG decode. Loaded into the same container for
    # intra-process zero-copy delivery to the fusion node's image_callback.
    fast_image_pub = ComposableNode(
        package='sample_midas_yolo_parallel_cpp',
        plugin='sample_midas_yolo_parallel_cpp::FastImagePublisherNode',
        name='fast_image_publisher_node',
        parameters=[{
            'filename': LaunchConfiguration('image_path'),
            'rate': LaunchConfiguration('publish_rate'),
        }],
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
            'yolo_tensor_data_type': 0,
            'yolo_pack_uint16_input': False,
            'midas_input_size': [256, 256],
            'yolo_input_size': [640, 640],
            'score_thresh': 0.25,
            'iou_thresh': 0.45,
        }],
    )

    # All nodes in one container: image publisher + inference + fusion.
    # Intra-process zero-copy for image_raw, tensor inputs, and tensor outputs.
    inference_container = ComposableNodeContainer(
        name='parallel_inference_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[fast_image_pub, shared_inference, fusion_cpp],
        output='screen',
    )

    return LaunchDescription([
        image_path_arg,
        combined_model_arg,
        midas_graph_name_arg,
        yolo_graph_name_arg,
        pub_rate_arg,
        inference_container,
    ])
