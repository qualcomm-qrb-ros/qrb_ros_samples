#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from audio_process import AudioProcessor
import threading


class SpeechRecognition(Node):
    def __init__(self):
        super().__init__("SpeechRecognition")
        self.publisher_ = self.create_publisher(String, "whisper_text", 10)

        # Dynamically Set Parameters via Topic
        self.subscription = self.create_subscription(
            Bool, "whisper_enable", self.set_whisper_enable_callback, 10
        )
        self.subscription

        # Declare Parameters
        self.declare_parameter("AudioEnergyThreshold", 0.5)
        self.declare_parameter("ShortTermWindow", 0.1)
        self.declare_parameter("MovingAverageWindow", 30)
        self.declare_parameter("AvailableWindow", 1)
        self.declare_parameter("LocalTiny", 1)
        self.declare_parameter('ServerUrl', 'http://10.92.128.242:5000/transcribe')

        # Get Parameters
        self.AudioEnergyThreshold = self.get_parameter("AudioEnergyThreshold").value
        self.ShortTermWindow = self.get_parameter("ShortTermWindow").value
        self.MovingAverageWindow = self.get_parameter("MovingAverageWindow").value
        self.AvailableWindow = self.get_parameter("AvailableWindow").value
        self.localtiny = self.get_parameter("LocalTiny").value
        self.server_url = self.get_parameter('ServerUrl').get_parameter_value().string_value

        # Initialize AudioProcessor
        self.audio_processor = AudioProcessor(
            server_url=self.server_url,
            audio_energy_threshold=self.AudioEnergyThreshold,
            short_term_window=self.ShortTermWindow,
            moving_average_window=self.MovingAverageWindow,
            available_window=self.AvailableWindow,
            localtiny=self.localtiny,
        )

        self.get_logger().info("Speech recognition initialization completed")
        if self.audio_processor.get_parameter_enable_asr():
            self.get_logger().info("Real-time speech recognition is enabled")
        else:
            self.get_logger().info("Real-time speech recognition is not enabled")

        self.running = True
        self.thread = threading.Thread(target=self.main_callback)
        self.thread.start()

    def set_whisper_enable_callback(self, msg):
        self.audio_processor.set_parameter_enable_asr(msg.data)
        if self.audio_processor.get_parameter_enable_asr():
            self.get_logger().info("Enabling real-time speech recognition...")
        else:
            self.get_logger().info("Disabling real-time speech recognition...")

    def main_callback(self):
        while self.running:
            self.audio_processor.audio_event.wait()
            if not self.running:
                break
            self.audio_processor.audio_event.clear()
            text = self.audio_processor.transcription_whisper()
            if text is None:
                continue
            msg = String()
            msg.data = text
            self.publisher_.publish(msg)

    def stop(self):
        self.running = False
        self.audio_processor.audio_event.set()


def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognition()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.thread.join()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
