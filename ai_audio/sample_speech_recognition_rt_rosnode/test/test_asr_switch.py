#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool  # 导入 Bool 消息类型

class SetEnableAsrPublisher(Node):
    def __init__(self):
        super().__init__('set_enable_asr_publisher')
        self.publisher_ = self.create_publisher(Bool, 'whisper_enable', 10)

    def send_message(self, enable):
        msg = Bool()
        msg.data = enable
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    publisher_node = SetEnableAsrPublisher()

    try:
        while rclpy.ok():
            user_input = input("Enter 'True' to enable ASR or 'False' to disable ASR: ")
            if user_input.lower() == 'true':
                publisher_node.send_message(True)
            elif user_input.lower() == 'false':
                publisher_node.send_message(False)
            else:
                print("Invalid input. Please enter 'True' or 'False'.")
            rclpy.spin_once(publisher_node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
        main()
