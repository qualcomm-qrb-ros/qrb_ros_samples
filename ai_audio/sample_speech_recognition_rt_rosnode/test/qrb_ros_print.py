#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class WhisperSubscriber(Node):

    def __init__(self):
        super().__init__('whisper_subscriber')
        self.subscription = self.create_subscription(
            String,
            'whisper_text',
            self.listener_callback,
            10)
        self.subscription  # 防止未使用变量的警告

    def listener_callback(self, msg):
        self.get_logger().info('收到的消息: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    whisper_subscriber = WhisperSubscriber()
    rclpy.spin(whisper_subscriber)
    whisper_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

