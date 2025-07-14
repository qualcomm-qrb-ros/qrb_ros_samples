'''
Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from sensor_msgs.msg import Image
import argparse

class NodePublisher(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("publish image %s" %name)

def main(args=None):
    rclpy.init(args=None)
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default="test")
    parser.add_argument('--picture', type=str, default="/tmp/data.png")
    args = parser.parse_args()
    node = NodePublisher("ocr_testnode")
    image_pub = node.create_publisher(Image, args.topic , 10)
    bridge = CvBridge()
    while True:
        color_frame = cv2.imread(args.picture)
        # publish image message
        img_msg = bridge.cv2_to_imgmsg(color_frame, encoding="bgr8")
        image_pub.publish(img_msg)

if __name__ == "__main__":
    main()
