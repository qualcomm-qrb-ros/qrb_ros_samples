# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='qrb_ros_speech_recognition_rt_rosnode',
            executable='qrb_ros_speech_recognition',
            name='SpeechRecognition',
            output='screen',
            parameters=[
                {'AudioEnergyThreshold': 0.5},  # Threshold for audio energy
                {'ShortTermWindow': 0.1},       # Time window (in seconds) for sampling audio energy
                {'MovingAverageWindow': 30},    # Number of samples for moving average window
                {'AvailableWindow': 1},         # Minimum valid audio length (in seconds)
                {'LocalTiny': 0},               # 1: run local tiny_en model; 0: use remote service
                {'ServerUrl': 'http://10.92.128.242:5000/transcribe'}  # Remote Service Sddress
            ]
        ),
    ])