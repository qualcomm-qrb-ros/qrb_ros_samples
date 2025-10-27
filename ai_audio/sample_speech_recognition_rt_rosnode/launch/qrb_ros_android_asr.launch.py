# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='qrb_ros_speech_recognition',
            executable='qrb_ros_android_asr',
            name='SpeechRecognition',
            output='screen',
        ),
    ])