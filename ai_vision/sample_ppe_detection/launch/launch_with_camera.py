# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Launch PPE detection from a live camera: USB (V4L2) or GMSL.
#
#   image source : usb_cam (ros-jazzy-usb-cam)      ->  /image_raw          [camera_type:=usb, default]
#               or qrb_ros_camera (ComposableNode)   ->  /cam<camera_id>_<stream_name>  [camera_type:=gmsl]
#   inference    : qrb_ros_nn_inference (NPU / HTP, ComposableNode)
#   post-process : ppe_detection_node  ->  /ppe_detection/{image,result}
#
# NOTE: the camera pushes frames at its configured rate, but NPU inference is
#       serial. ppe_detection_node self-throttles: while a frame is still
#       being inferred it drops incoming frames, so effective throughput
#       matches the NPU. Lower the capture rate to reduce wasted CPU decoding.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ── Camera selection ─────────────────────────────────────────────────
    camera_type_arg = DeclareLaunchArgument(
        'camera_type',
        default_value='usb',
        description='Camera source to use: "usb" (V4L2 via usb_cam) or '
                    '"gmsl" (via qrb_ros_camera).'
    )

    # ── Camera arguments (forwarded to usb_cam) ─────────────────────────
    video_device_arg = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/video0',
        description='V4L2 device node of the USB camera (e.g. /dev/video0). '
                    'Only used when camera_type:=usb.'
    )

    framerate_arg = DeclareLaunchArgument(
        'framerate',
        default_value='30.0',
        description='USB camera capture rate (Hz). ppe_detection_node drops frames '
                    'it cannot keep up with; lower this to reduce CPU decoding. '
                    'Only used when camera_type:=usb.'
    )

    # yuyv2rgb: widest webcam compatibility (esp. 640x480).
    # mjpeg2rgb: needed by most webcams for 720p/1080p at high fps.
    pixel_format_arg = DeclareLaunchArgument(
        'pixel_format',
        default_value='yuyv2rgb',
        description='usb_cam pixel format. Use "mjpeg2rgb" for 720p/1080p webcams. '
                    'Output is rgb8, as expected by ppe_detection_node. '
                    'Only used when camera_type:=usb.'
    )

    # ── Camera arguments (forwarded to qrb_ros_camera) ──────────────────
    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='GMSL camera index exposed by qrb_ros_camera. '
                    'Only used when camera_type:=gmsl.'
    )

    stream_name_arg = DeclareLaunchArgument(
        'stream_name',
        default_value='stream1',
        description='qrb_ros_camera stream name. Determines the published '
                    'topic: /cam<camera_id>_<stream_name>. '
                    'Only used when camera_type:=gmsl.'
    )

    camera_fps_arg = DeclareLaunchArgument(
        'camera_fps',
        default_value='30',
        description='GMSL camera capture rate (Hz). '
                    'Only used when camera_type:=gmsl.'
    )

    camera_info_file_arg = DeclareLaunchArgument(
        'camera_info_file',
        default_value='camera_info_OX03F10_yuv.yaml',
        description='camera_info yaml filename under the qrb_ros_camera '
                    'package share/config directory. Ships with '
                    'camera_info_{imx577,ov9282,ar0231,OX03F10_yuv,OX03F10_bayer}.yaml; '
                    'default matches the OX03F10 GMSL module used on the '
                    'IQ-9075/IQ-8275 EVKs. Only used when camera_type:=gmsl.'
    )

    # ── Shared capture resolution ────────────────────────────────────────
    image_width_arg = DeclareLaunchArgument(
        'image_width',
        default_value='640',
        description='Capture width in pixels.'
    )

    image_height_arg = DeclareLaunchArgument(
        'image_height',
        default_value='480',
        description='Capture height in pixels.'
    )

    # ── Model / backend arguments ───────────────────────────────────────
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/opt/model/gear_guard_net_ctx.bin',
        description='Path to the PPE detection model. Must be a precompiled QNN '
                    'context binary (.bin), NOT the raw .dlc file.'
    )

    backend_arg = DeclareLaunchArgument(
        'backend',
        default_value='/usr/lib/libQnnHtp.so',
        description='QNN backend library (HTP for NPU).'
    )

    # ── Detection / output arguments (forwarded to ppe_detection_node) ───
    conf_thresh_arg = DeclareLaunchArgument(
        'conf_thresh',
        default_value='0.5',
        description='Confidence threshold for detections.'
    )

    iou_thresh_arg = DeclareLaunchArgument(
        'iou_thresh',
        default_value='0.5',
        description='IoU threshold for per-class NMS.'
    )

    box_hold_frames_arg = DeclareLaunchArgument(
        'box_hold_frames',
        default_value='5',
        description='Keep last detected box for N frames to reduce flicker. 0 = off.'
    )

    save_path_arg = DeclareLaunchArgument(
        'save_path',
        default_value='',
        description='If set, overwrite-save the latest annotated frame to this image file.'
    )

    save_video_path_arg = DeclareLaunchArgument(
        'save_video_path',
        default_value='',
        description='If set, record annotated frames into this video file (.avi/MJPG recommended).'
    )

    output_fps_arg = DeclareLaunchArgument(
        'output_fps',
        default_value='2.0',
        description='Frame rate of the recorded demo video (match effective throughput).'
    )

    # ── Launch Configurations ───────────────────────────────────────────
    camera_type  = LaunchConfiguration('camera_type')
    video_device = LaunchConfiguration('video_device')
    framerate    = LaunchConfiguration('framerate')
    pixel_format = LaunchConfiguration('pixel_format')
    camera_id        = LaunchConfiguration('camera_id')
    stream_name      = LaunchConfiguration('stream_name')
    camera_fps       = LaunchConfiguration('camera_fps')
    camera_info_file = LaunchConfiguration('camera_info_file')
    image_width  = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')
    model_path   = LaunchConfiguration('model_path')
    backend      = LaunchConfiguration('backend')
    conf_thresh     = LaunchConfiguration('conf_thresh')
    iou_thresh      = LaunchConfiguration('iou_thresh')
    box_hold_frames = LaunchConfiguration('box_hold_frames')
    save_path       = LaunchConfiguration('save_path')
    save_video_path = LaunchConfiguration('save_video_path')
    output_fps      = LaunchConfiguration('output_fps')

    is_usb  = IfCondition(PythonExpression(["'", camera_type, "' == 'usb'"]))
    is_gmsl = IfCondition(PythonExpression(["'", camera_type, "' == 'gmsl'"]))

    namespace = 'ppe_detection_container'

    # ── USB Camera Node (image source) ──────────────────────────────────
    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        condition=is_usb,
        parameters=[{
            'video_device': video_device,
            'image_width':  image_width,
            'image_height': image_height,
            'framerate':    framerate,
            'pixel_format': pixel_format,
            'camera_name':  'usb_cam',
            'frame_id':     'usb_cam',
        }],
        remappings=[
            ('image_raw', '/image_raw'),
        ]
    )

    # ── qrb_ros_camera Node (image source, GMSL) ────────────────────────
    camera_info_config_file_path = PathJoinSubstitution([
        get_package_share_directory('qrb_ros_camera'),
        'config', camera_info_file])

    gmsl_camera_node = ComposableNode(
        package='qrb_ros_camera',
        plugin='qrb_ros::camera::CameraNode',
        name='camera_node',
        parameters=[{
            'camera_id': camera_id,
            'stream_size': 1,
            'stream_name': [stream_name],
            'stream1': {
                'height': image_height,
                'width': image_width,
                'fps': camera_fps,
            },
            'camera_info_path': camera_info_config_file_path,
        }]
    )

    gmsl_camera_container = ComposableNodeContainer(
        name='camera_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        output='screen',
        condition=is_gmsl,
        composable_node_descriptions=[gmsl_camera_node],
    )

    # ── QNN Inference Node (ComposableNode) ─────────────────────────────
    nn_inference_node = ComposableNode(
        package='qrb_ros_nn_inference',
        namespace=namespace,
        plugin='qrb_ros::nn_inference::QrbRosInferenceNode',
        name='nn_inference_node',
        parameters=[{
            'backend_option': backend,
            'model_path': model_path,
            'log_level': 'info',
        }]
    )

    # ── Container for ComposableNodes ───────────────────────────────────
    container = ComposableNodeContainer(
        name='ppe_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container',
        output='screen',
        composable_node_descriptions=[nn_inference_node],
        sigterm_timeout='3',
        sigkill_timeout='5'
    )

    # ── PPE Detection Node ──────────────────────────────────────────────
    # image_raw remap: '/image_raw' when USB (identity), '/cam<id>_<stream>' when GMSL.
    image_raw_topic = PythonExpression([
        "'/image_raw' if '", camera_type, "' == 'usb' else "
        "'/cam' + '", camera_id, "' + '_' + '", stream_name, "'"
    ])

    ppe_detection_node = Node(
        package='sample_ppe_detection',
        executable='ppe_detection_node',
        name='ppe_detection_node',
        namespace=namespace,
        output='screen',
        parameters=[{
            'conf_thresh':     conf_thresh,
            'iou_thresh':      iou_thresh,
            'box_hold_frames': box_hold_frames,
            'save_path':       save_path,
            'save_video_path': save_video_path,
            'output_fps':      output_fps,
        }],
        remappings=[
            ('/image_raw', image_raw_topic),
            ('qrb_inference_input_tensor',
             '/' + namespace + '/qrb_inference_input_tensor'),
            ('qrb_inference_output_tensor',
             '/' + namespace + '/qrb_inference_output_tensor'),
        ]
    )

    return LaunchDescription([
        camera_type_arg,
        video_device_arg,
        image_width_arg,
        image_height_arg,
        framerate_arg,
        pixel_format_arg,
        camera_id_arg,
        stream_name_arg,
        camera_fps_arg,
        camera_info_file_arg,
        model_path_arg,
        backend_arg,
        conf_thresh_arg,
        iou_thresh_arg,
        box_hold_frames_arg,
        save_path_arg,
        save_video_path_arg,
        output_fps_arg,
        LogInfo(msg=['   Starting PPE Detection with camera_type=', camera_type]),
        LogInfo(condition=is_usb, msg=['   Device: ', video_device,
                     '  (', image_width, 'x', image_height, ' @ ', framerate, ' Hz, ', pixel_format, ')']),
        LogInfo(condition=is_gmsl, msg=['   Camera: id=', camera_id, ' stream=', stream_name,
                     '  (', image_width, 'x', image_height, ' @ ', camera_fps, ' Hz)']),
        LogInfo(msg=['   Model : ', model_path]),
        usb_cam_node,
        gmsl_camera_container,
        container,
        ppe_detection_node,
    ])
