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
from ament_index_python.packages import get_package_share_directory
import shutil
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList
import struct
from std_msgs.msg import String


class ResNet101PostProcessNode(Node):
    def __init__(self):
        super().__init__('resnet101_postprocess_node')
        self.subscriber = self.create_subscription(
            TensorList,
            'qrb_inference_output_tensor',  # subscriber topic
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(
            String,
            'resnet101_output',  # publish topic
            10
        )

        self.bridge = CvBridge()
        self.get_logger().info(f'Initial ROS Node resnet101')
        
        self.declare_parameter('model_path', '/opt/model/')
        
        self.model_path = self.get_parameter('model_path').value
        
        self.get_logger().info(f'model path: {self.model_path}')

        #read output label
        with open(self.model_path + "imagenet_labels.txt", "r") as file:
            self.class_names = file.readlines()    
                  
    def image_callback(self, msg):
        try:
            for result_tensor in msg.tensor_list:  # search tensor_list
                #self.get_logger().info(f'result_tensor shape is {result_tensor.shape}')
                output_data = np.array(result_tensor.data)
                predicted_class = np.argmax(output_data)
               
                class_id = self.class_names[predicted_class]
                self.publisher.publish(String(data=class_id))
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    
    rclpy.init(args=args)
    node = ResNet101PostProcessNode() 
       
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
