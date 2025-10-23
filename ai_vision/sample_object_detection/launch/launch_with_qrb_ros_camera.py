# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    default_label_file = "/opt/coco8.yaml"
    #default_model_path = "/work/model/yolov8_det_qcs6490.tflite"
    default_model_path = "/opt/model/yolov8_det_qcs9075.bin"
    default_nms_iou_thres = "0.5"
    default_nms_score_thres = "0.7"
    default_target_res = "640x640"
    default_tensor_fmt = "nhwc"
    default_normalize = "True"
    defautl_data_type = "float32"
    defautl_cam_fps = "10"

    default_cam_fps_arg = DeclareLaunchArgument(
        "cam_fps",
        default_value=defautl_cam_fps,
        description="camera fps",
    )

    default_target_res_arg = DeclareLaunchArgument(
        "target_res",
        default_value=default_target_res,
        description="resolution required by model",
    )

    normalize_arg = DeclareLaunchArgument(
        "normalize",
        default_value=default_normalize,
        description="whether need normalize",
    )

    tensor_fmt_arg = DeclareLaunchArgument(
        "tensor_fmt",
        default_value=default_tensor_fmt,
        description="nhwc or nchw",
    )

    data_type_arg = DeclareLaunchArgument(
        "data_type",
        default_value=defautl_data_type,
        description="flaot32 float64 uint8",
    )

    label_file_arg = DeclareLaunchArgument(
        "label_file",
        default_value=default_label_file,
        description="label files for yolov8 model",
    )

    model_file_arg = DeclareLaunchArgument(
        "model",
        default_value=default_model_path,
        description="YOLOv8 detection model file path",
    )

    score_thres_arg = DeclareLaunchArgument(
        "score_thres",
        default_value=default_nms_score_thres,
        description="score(confidence) threshold value, between 0.0 ~ 1.0",
    )

    iou_thres_arg = DeclareLaunchArgument(
        "iou_thres",
        default_value=default_nms_iou_thres,
        description="iou threshold value, between 0.0 ~ 1.0",
    )

    #qrb ros camera
    camera_info_config_file_path =PathJoinSubstitution([
        get_package_share_directory('qrb_ros_camera'),
        'config', 'camera_info_imx577.yaml'])
    camera_node = ComposableNode(
        package='qrb_ros_camera',
        namespace="",
        plugin='qrb_ros::camera::CameraNode',
        name='camera_node',
        parameters=[{
            'camera_id': 0,
            'stream_size': 1,
            'stream_name': ["stream1"],
            'stream1':{
                'height':480,
                'width':640,
                'fps':30,
            },
            'camera_info_path': camera_info_config_file_path,
            'dump': False,
            'dump_camera_info_': False,
        }]
    )   
    
    ## sub: /input_image
    ## pub: /encoded_image
    ## pub: /resized_image
    preprocess_node = ComposableNode(
        package="qrb_ros_cv_tensor_common_process",
        plugin="qrb_ros::cv_tensor_common_process::CvTensorCommonProcessNode",
        name="yolo_preprocess_node",
        parameters=[
            {"target_res": LaunchConfiguration("target_res")},
            {"normalize": LaunchConfiguration("normalize")},
            {"tensor_fmt": LaunchConfiguration("tensor_fmt")},
            {"data_type": LaunchConfiguration("data_type")},
        ],
        remappings = [
            ("input_image", "/cam0_stream1"),
            ("encoded_image", "qrb_inference_input_tensor"),
        ],
    )

    ## sub: /qrb_inference_input_tensor
    ## pub: /output_tensor (remap to /yolo_detect_tensor_output)
    inference_node = ComposableNode(
        package='qrb_ros_nn_inference',
        plugin='qrb_ros::nn_inference::QrbRosInferenceNode',
        name='nn_inference_node',
        parameters=[{
            #'backend_option': "/usr/lib/unsigned/libQnnHtpV73.so",
            'backend_option': "/usr/lib/libQnnHtp.so",
            'model_path': LaunchConfiguration("model"),
        },
        ],
        remappings=[
            ('qrb_inference_output_tensor', 'yolo_detect_tensor_output'),
        ]
    )

    ## sub: /yolo_detect_tensor_output
    ## pub: /yolo_detect_result
    postprocess_node = ComposableNode(
        package="qrb_ros_yolo_process",
        plugin="qrb_ros::yolo_process::YoloDetPostProcessNode",
        name="yolo_detection_postprocess_node",
        parameters=[
            {"label_file": LaunchConfiguration("label_file")},
            {"score_thres": LaunchConfiguration("score_thres")},
            {"iou_thres": LaunchConfiguration("iou_thres")},
        ],
    )

    ## sub:
    ##   - /yolo_detect_result
    ##   - /resized_image
    ## pub: /yolo_detect_overlay
    overlay_node = ComposableNode(
        package="qrb_ros_yolo_process",
        plugin="qrb_ros::yolo_process::YoloDetOverlayNode",
        name="yolo_detection_overlay_node",
        parameters=[
            {"target_res": LaunchConfiguration("target_res")},
        ],
    )

    container = ComposableNodeContainer(
        name="yolo_node_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[camera_node,overlay_node, postprocess_node, inference_node, preprocess_node],
        output="screen",
    )
    return LaunchDescription(
        [
            default_cam_fps_arg,
            label_file_arg,
            model_file_arg,
            score_thres_arg,
            iou_thres_arg,
            default_target_res_arg,
            normalize_arg,
            tensor_fmt_arg,
            data_type_arg,
            container,
        ]
    )

