# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import Node


def generate_launch_description():
    default_label_file = "/opt/coco8.yaml"
    default_model_path = "/opt/model/yolov8_det_qcs9075.bin"
    default_nms_iou_thres = "0.5"
    default_nms_score_thres = "0.7"
    default_target_res = "640x640"
    default_tensor_fmt = "nhwc"
    default_normalize = "True"
    defautl_data_type = "float32"

    default_cam_fps_arg = DeclareLaunchArgument(
        "cam_fps",
        default_value="10",
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
        default_value='10.0',
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

    video_device = LaunchConfiguration('video_device')
    pixel_format = LaunchConfiguration('pixel_format')
    framerate = LaunchConfiguration('framerate')
    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')

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
            ("input_image", "/image_raw"),
            ("encoded_image", "qrb_inference_input_tensor"),
        ],
    )

    inference_node = ComposableNode(
        package='qrb_ros_nn_inference',
        plugin='qrb_ros::nn_inference::QrbRosInferenceNode',
        name='nn_inference_node',
        parameters=[{
            'backend_option': "/usr/lib/libQnnHtp.so",
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
        composable_node_descriptions=[overlay_node, postprocess_node, inference_node, preprocess_node],
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
            video_device_arg,
            pixel_format_arg,
            framerate_arg,
            image_width_arg,
            image_height_arg,
            usb_cam_node,
            container,
        ]
    )
