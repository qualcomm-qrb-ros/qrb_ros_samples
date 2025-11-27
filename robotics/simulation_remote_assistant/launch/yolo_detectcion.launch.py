# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer

def generate_launch_description():
    default_label_file = ""
    default_model_path = ""
    default_backend_option = ""
    #default_model_path = "/opt/model/yolov8_det_qcs6490.tflite"
    #default_backend_option = "gpu"
    default_nms_iou_thres = "0.5"
    default_nms_score_thres = "0.5"
    default_target_res = "640x640"
    default_tensor_fmt = "nhwc"
    default_normalize = "True"
    default_data_type = "float32"

    backend_option_arg = DeclareLaunchArgument(
        "backend_option",
        default_value=default_backend_option,
        description="backend option for inference",
    )

    model_file_arg = DeclareLaunchArgument(
        "model",
        default_value=default_model_path,
        description="YOLOv8 detection model file path",
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
        default_value=default_data_type,
        description="float32 float64 uint8",
    )

    label_file_arg = DeclareLaunchArgument(
        "label_file",
        default_value=default_label_file,
        description="label files for yolov8 model",
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
        remappings=[
            ("input_image", "/camera/color/image_raw"),
            ("encoded_image", "qrb_inference_input_tensor"),
        ],
    )

    inference_node = ComposableNode(
        package='qrb_ros_nn_inference',
        plugin='qrb_ros::nn_inference::QrbRosInferenceNode',
        name='nn_inference_node',
        parameters=[{
            'backend_option': LaunchConfiguration("backend_option"),
            'model_path': LaunchConfiguration("model"),
        }],
        remappings=[
            ('qrb_inference_output_tensor', 'yolo_detect_tensor_output'),
        ]
    )

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
        composable_node_descriptions=[
            overlay_node, postprocess_node, inference_node, preprocess_node
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            label_file_arg,
            model_file_arg,
            backend_option_arg,    
            score_thres_arg,
            iou_thres_arg,
            default_target_res_arg,
            normalize_arg,
            tensor_fmt_arg,
            data_type_arg,
            container,
        ]
    )
