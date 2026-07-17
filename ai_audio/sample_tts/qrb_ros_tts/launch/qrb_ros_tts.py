# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    pkg_share = get_package_share_directory('qrb_ros_tts')
    default_cfg = os.path.join(pkg_share, 'config', 'tts_paths.cfg')

    cfg_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_cfg,
        description='Path to TTS paths config file (key=value format)',
    )

    wav_arg = DeclareLaunchArgument(
        'output_wav',
        default_value='output.wav',
        description='Path to save synthesised WAV file',
    )

    speaker = ComposableNode(
        package='qrb_ros_tts',
        plugin='SpeakerNode',
        name='speaker_node',
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'output_wav':  LaunchConfiguration('output_wav'),
        }],
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    container = ComposableNodeContainer(
        name='speaker_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[speaker],
        output='screen',
    )

    return LaunchDescription([
        cfg_arg,
        wav_arg,
        container,
    ])
