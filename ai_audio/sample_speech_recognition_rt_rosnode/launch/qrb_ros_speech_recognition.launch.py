# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='qrb_ros_speech_recognition',
            executable='qrb_ros_speech_recognition.py',
            name='SpeechRecognition',
            output='screen',
            parameters=[
                {'AudioEnergyThreshold': 0.5}, # 音频强度的阈值
                {'ShortTermWindow': 0.1}, # 计算音频强度的采样时间 s
                {'MovingAverageWindow': 30}, # 累计多少个采样强度的窗口长度
                {'AvailableWindow': 1}, # 最小有效音频长度限制 s
                {'LocalTiny': 1} # 1：运行本地 tiny_en 模型 0:远程服务器大模型
            ]
        ),
    ])