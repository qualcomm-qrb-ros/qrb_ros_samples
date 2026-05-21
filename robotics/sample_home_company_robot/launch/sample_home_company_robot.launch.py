# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ==============================
    # Defaults
    # ==============================
    default_label_file = "/opt/model/coco8.yaml"
    default_yolo_model_path = "/opt/model/yolov8_det_qcs9075.bin"
    default_reid_model_path = "/opt/model/osnet.bin"
    default_backend_option = "/usr/lib/libQnnHtp.so"
    default_nms_iou_thres = "0.5"
    default_nms_score_thres = "0.7"
    default_target_res = "640x640"
    default_tensor_fmt = "nhwc"
    default_normalize = "True"
    default_data_type = "float32"

    follow_me_pkg_dir = get_package_share_directory("follow_me")
    tracking_config_file = os.path.join(follow_me_pkg_dir, "config", "tracking_params.yaml")

    # ==============================
    # Launch Arguments
    # ==============================
    declare_namespace = DeclareLaunchArgument(
        "namespace", default_value="", description="Namespace for nodes"
    )
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation time"
    )
    declare_log_level = DeclareLaunchArgument(
        "log_level",
        default_value="INFO",
        description="Log level (DEBUG, INFO, WARN, ERROR, FATAL)",
    )

    target_res_arg = DeclareLaunchArgument(
        "target_res", default_value=default_target_res, description="Model input resolution"
    )
    normalize_arg = DeclareLaunchArgument(
        "normalize", default_value=default_normalize, description="Enable normalization"
    )
    tensor_fmt_arg = DeclareLaunchArgument(
        "tensor_fmt", default_value=default_tensor_fmt, description="nhwc or nchw"
    )
    data_type_arg = DeclareLaunchArgument(
        "data_type", default_value=default_data_type, description="float32/float64/uint8"
    )
    label_file_arg = DeclareLaunchArgument(
        "label_file", default_value=default_label_file, description="YOLO labels file"
    )
    yolo_model_arg = DeclareLaunchArgument(
        "yolo_model", default_value=default_yolo_model_path, description="YOLO model path"
    )
    score_thres_arg = DeclareLaunchArgument(
        "score_thres",
        default_value=default_nms_score_thres,
        description="NMS score threshold [0.0, 1.0]",
    )
    iou_thres_arg = DeclareLaunchArgument(
        "iou_thres",
        default_value=default_nms_iou_thres,
        description="NMS IoU threshold [0.0, 1.0]",
    )

    reid_model_arg = DeclareLaunchArgument(
        "reid_model_path",
        default_value=default_reid_model_path,
        description="Path to ReID model file",
    )
    backend_option_arg = DeclareLaunchArgument(
        "backend_option",
        default_value=default_backend_option,
        description="QNN backend library path",
    )

    # ==============================
    # YOLO Pipeline (Composable)
    # ==============================
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
            ("input_image", "/color/preview/image"),
            ("encoded_image", "qrb_inference_input_tensor"),
        ],
    )

    inference_node = ComposableNode(
        package="qrb_ros_nn_inference",
        plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
        name="nn_inference_node",
        parameters=[
            {
                "backend_option": LaunchConfiguration("backend_option"),
                "model_path": LaunchConfiguration("yolo_model"),
            }
        ],
        remappings=[("qrb_inference_output_tensor", "yolo_detect_tensor_output")],
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
        parameters=[{"target_res": LaunchConfiguration("target_res")}],
    )

    yolo_container = ComposableNodeContainer(
        name="yolo_node_container",
        namespace=LaunchConfiguration("namespace"),
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            overlay_node,
            postprocess_node,
            inference_node,
            preprocess_node,
        ],
        output="screen",
    )

    # ==============================
    # ReID Pipeline (Composable)
    # ==============================
    people_reid_node = ComposableNode(
        package="qrb_ros_people_reid",
        plugin="qrb_ros::people_reid::PeopleReIDNode",
        name="people_reid_node",
        parameters=[],
    )

    reid_inference_node = ComposableNode(
        package="qrb_ros_nn_inference",
        plugin="qrb_ros::nn_inference::QrbRosInferenceNode",
        name="reid_inference_node",
        parameters=[
            {
                "backend_option": LaunchConfiguration("backend_option"),
                "model_path": LaunchConfiguration("reid_model_path"),
            }
        ],
        remappings=[
            ("qrb_inference_input_tensor", "/reid_inference_input_tensor"),
            ("qrb_inference_output_tensor", "/reid_tensor_output"),
        ],
    )

    reid_container = ComposableNodeContainer(
        name="people_reid_container",
        namespace=LaunchConfiguration("namespace"),
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[people_reid_node, reid_inference_node],
        output="screen",
    )

    # ==============================
    # Follow Me tracker
    # ==============================
    person_tracker_node = Node(
        package="follow_me",
        executable="person_tracker_node",
        name="person_tracker_node",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[
            tracking_config_file,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=[
            "--ros-args",
            "--log-level",
            ["person_tracker_node:=", LaunchConfiguration("log_level")],
        ],
    )

    # ==============================
    # BT ROS Node (from sample_home_company_robot)
    # ==============================
    bt_ros_node = Node(
        package="sample_home_company_robot",
        executable="bt_ros_node",
        name="bt_ros_node",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=[
            "--ros-args",
            "--log-level",
            ["bt_ros_node:=", LaunchConfiguration("log_level")],
        ],
    )

    return LaunchDescription(
        [
            declare_namespace,
            declare_use_sim_time,
            declare_log_level,
            target_res_arg,
            normalize_arg,
            tensor_fmt_arg,
            data_type_arg,
            label_file_arg,
            yolo_model_arg,
            score_thres_arg,
            iou_thres_arg,
            reid_model_arg,
            backend_option_arg,
            yolo_container,
            reid_container,
            person_tracker_node,
            bt_ros_node,
        ]
    )
