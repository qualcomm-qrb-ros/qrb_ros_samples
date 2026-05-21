#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_msgs.msg import ByteMultiArray
import time
import numpy as np
import io
import wave
import os
import requests

class SpeechRecognition(Node):
    def __init__(self):
        super().__init__("SpeechRecognition")
        self.publisher_ = self.create_publisher(String, "whisper_text", 10)

        self.subscription = self.create_subscription(
            ByteMultiArray,
            'audio_file',
            self.audio_callback,
            10
        )

        self.get_logger().info(f'🎧 waiting for audio files ......')

        self.declare_parameter('ServerUrl', 'http://10.92.128.242:5000/transcribe')
        self.server_url = self.get_parameter('ServerUrl').get_parameter_value().string_value

    def audio_callback(self, msg):
        self.get_logger().info(f'reveived the wav')
        try:
            byte_data = bytes([b[0] if isinstance(b, bytes) else b for b in msg.data])
            with wave.open(io.BytesIO(byte_data), 'rb') as wav_file:
                num_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                num_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(num_frames)

                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                if num_channels > 1:
                    audio_np = audio_np.reshape(-1, num_channels).mean(axis=1)
                audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max

            transcription = self.transcription_to_server(audio_np)
        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

        if transcription is None:
            self.get_logger().info(f'Transcription: NA')
        else:
            self.get_logger().info(f'Transcription: {transcription}')

    def transcription_to_server(self, audio_data):
        self.get_logger().info(f"Performing speech recognition")
        start_time = time.time()
        try:
            self.get_logger().info("service module")
            transcription = self.send_audio_to_server(audio_data)
        except Exception as e:
            self.get_logger().info(f"An error occurred during speech recognition: {e}")
            transcription = "NA"
        elapsed_time = time.time() - start_time
        self.get_logger().info(f"[Total elapsed time: {elapsed_time} seconds]")

        msg = String()
        msg.data = transcription
        self.publisher_.publish(msg)
        
        return transcription

    def send_audio_to_server(self, audio_data):
        try:
            self.get_logger().info(f"send audio to server")
            files = {"audio": audio_data.tobytes()}
            response = requests.post(self.server_url, files=files)
            if response.status_code == 200:
                result = response.json()
            else:
                print(f"Request failed: {response.status_code}")
                result = {"text": ""}
        except Exception as e:
            print(f"An error occurred while sending the request: {e}")
            result = {"text": ""}

        return result["text"]

    def stop(self):
        self.running = False
        # self.audio_event.set()

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
