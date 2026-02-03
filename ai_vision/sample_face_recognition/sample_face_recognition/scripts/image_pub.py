# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Copyright (c) 2026, Qualcomm Innovation Center, Inc. and/or its affiliates.
# All rights reserved.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class ImageOncePublisher(Node):
    def __init__(self, image_path: Path, topic: str, rate_hz: float, frame_id: str, encoding: str):
        super().__init__('image_once_publisher')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )

        self.pub = self.create_publisher(Image, topic, qos)
        self.bridge = CvBridge()

        img = cv2.imdecode(
            np_from_file(image_path),
            cv2.IMREAD_COLOR
        )
        if img is None:
            raise RuntimeError(f'Failed to read image: {image_path}')

        # OpenCV uses BGR channel order by default; commonly used encoding is 'bgr8
        self.msg = self.bridge.cv2_to_imgmsg(img, encoding=encoding)
        self.msg.header.frame_id = frame_id

        self.rate_hz = rate_hz
        if rate_hz <= 0:
            # Publish once
            self.publish_once_and_quit()
        else:
            # Publish in a loop
            period = 1.0 / rate_hz
            self.timer = self.create_timer(period, self.timer_cb)
            self.get_logger().info(f'Loop publishing at {rate_hz:.2f} Hz to {topic}')

    def publish_once_and_quit(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)
        self.get_logger().info('Image published once, exiting.')
        # Give DDS a short time to deliver the message
        time.sleep(0.05)
        rclpy.shutdown()

    def timer_cb(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)


def np_from_file(path: Path):
    import numpy as np
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description='Publish an image to /input_faces as sensor_msgs/msg/Image')
    parser.add_argument('--image', '-i', required=True, type=Path, help='Path to the image file')
    parser.add_argument('--topic', '-t', default='/input_faces', help='Topic name (default: /input_faces)')
    parser.add_argument('--rate', '-r', type=float, default=0.0,
                        help='Publish rate in Hz. 0 = publish once and exit (default: 0)')
    parser.add_argument('--frame-id', default='camera', help='Header frame_id (default: camera)')
    parser.add_argument('--encoding', default='bgr8', help="Image encoding, e.g. 'bgr8','rgb8','mono8' (default: bgr8)")
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    try:
        node = ImageOncePublisher(args.image, args.topic, args.rate, args.frame_id, args.encoding)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()