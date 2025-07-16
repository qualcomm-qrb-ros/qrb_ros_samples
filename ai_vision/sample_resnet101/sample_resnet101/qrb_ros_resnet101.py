# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
import numpy as np
import subprocess
import sys
import shutil
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList
import struct


class ResNet101Node(Node):
    def __init__(self):
        super().__init__('resnet101_node')
        self.subscriber = self.create_subscription(
            Image,
            'image_raw',  # subscriber topic
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(
            TensorList,
            'qrb_inference_input_tensor',  # publish topic
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info(f'Initial ROS Node resnet101')
        
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')            
            cv_image = cv2.resize(cv_image, (224, 224))
 
            img_data = cv_image.tobytes()
            msg = TensorList()
            #tensor list to input
            tensor = Tensor()
            tensor.data_type = 0
            tensor.name = "resnet101__input_tensor"
            tensor.shape = [1,224, 224,3]
            tensor.data = img_data
       
            msg.tensor_list.append(tensor)
            
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
         
    rclpy.init(args=args)
    node = ResNet101Node() 
       
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
