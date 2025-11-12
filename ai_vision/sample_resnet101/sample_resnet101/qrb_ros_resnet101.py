# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import cv2
import numpy as np
import subprocess
import sys
import shutil
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList
import struct
from typing import List
import numbers


class ResNet101Node(Node):
    """
    ResNet101Node subscribes to an Image topic and publishes TensorList messages
    for inference. Improvements added:
    - Checks node active state before publishing
    - Validates that the published object is a proper ROS message (TensorList)
    - Manages node lifecycle flag and ensures proper cleanup on destroy
    """

    def __init__(self):
        super().__init__('resnet101_node')
        self._active = True  # lifecycle flag: True while node is healthy / running
        self.bridge = CvBridge()

        # Subscriber
        self.subscriber = self.create_subscription(
            Image,
            'image_raw',  # subscriber topic
            self.image_callback,
            10
        )

        # Publisher
        self.publisher = self.create_publisher(
            TensorList,
            'qrb_inference_input_tensor',  # publish topic
            10
        )

        self.get_logger().info('Initialized ResNet101Node')

    def destroy_node(self):
        """
        Override destroy_node to manage lifecycle state and cleanup.
        """
        # mark inactive first to prevent further publishes
        self._active = False
        try:
            # remove subscriptions/publishers if needed (they will be cleaned up by rclpy)
            if hasattr(self, 'subscriber') and self.subscriber is not None:
                # If desired, we can explicitly destroy subscriptions, but rclpy handles this on node destruction.
                pass
            if hasattr(self, 'publisher') and self.publisher is not None:
                pass
        except Exception as e:
            print(f'Error during node cleanup: {e}')
        # call parent cleanup
        try:
            super().destroy_node()
        except Exception:
            # Some versions of rclpy may raise if multiple destroy calls happen; ignore safely
            pass

    def nv12_to_bgr(self, nv12_image: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Convert NV12 image buffer (1D uint8) to BGR image using OpenCV.
        """
        expected_len = width * height * 3 // 2
        if nv12_image.size != expected_len:
            raise ValueError(f"NV12 buffer length mismatch: expected {expected_len}, got {nv12_image.size}")
        yuv = nv12_image.reshape((height * 3 // 2, width))
        bgr_data = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        return bgr_data      

    def _can_publish(self) -> bool:
        """
        Determine if publishing is allowed: node active and ROS hasn't been shutdown.
        """
        if not self._active:
            #self.get_logger().warning('Node marked inactive; skipping publish.')
            return False
        if not rclpy.ok():
            # rclpy.ok() returns False when ROS is shutting down
            return False
        if not hasattr(self.publisher, 'publish') or not callable(getattr(self.publisher, 'publish')):
            return False
        return True

    def image_callback(self, msg: Image):
        """
        Callback executed on incoming image messages. Converts image to the
        expected size/format and publishes a TensorList for inference.
        """
        
        if not rclpy.ok():
            #self.get_logger().warning('Node not running; skipping publish.')
            print(f"Node not running; skipping publish.")
            return False
        try:
            # Decode incoming image with encoding checks
            if msg.encoding == 'nv12':
                # Convert raw bytes to numpy array
                nv12_data = np.frombuffer(msg.data, dtype=np.uint8)
                cv_image = self.nv12_to_bgr(nv12_data, msg.width, msg.height)
            elif msg.encoding == 'bgr8':
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                except CvBridgeError as cb_e:
                    self.get_logger().error(f'CvBridge failed to convert image: {cb_e}')
                    return
            elif msg.encoding == 'rgb8':
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    # convert RGB to BGR for OpenCV processing
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                except CvBridgeError as cb_e:
                    self.get_logger().error(f'CvBridge failed to convert image: {cb_e}')
                    return
            else:
                self.get_logger().error(f'Unsupported image encoding: {msg.encoding}')
                return

            # Basic sanity check for image shape
            if cv_image is None or cv_image.size == 0:
                self.get_logger().error('Converted image is empty.')
                return

            # Resize to model input
            cv_image = cv2.resize(cv_image, (224, 224))

            # Prepare tensor data
            img_data = cv_image.tobytes()

            out_msg = TensorList()
            tensor = Tensor()
            tensor.data_type = 0
            tensor.name = "resnet101__input_tensor"
            tensor.shape = [1, 224, 224, 3]
            tensor.data = img_data

            out_msg.tensor_list.append(tensor)

            # Before publishing, ensure node is running and object is valid ROS message
            if not self._can_publish():
                # Do not attempt to publish if node is not active or rclpy is shutting down
                return

            # final safety: ensure out_msg is of correct type
            if not isinstance(out_msg, TensorList):
                self.get_logger().error('Constructed message is not a TensorList instance; aborting publish.')
                return

            # Publish safely
            try:
                self.publisher.publish(out_msg)
            except Exception as e:
                # Catch publish errors (e.g., if publisher was already destroyed concurrently)
                print(f'Failed to publish TensorList: {e}')

        except Exception as e:
            # Log unexpected exceptions but keep node alive if possible
            print(f'Error processing image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ResNet101Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('KeyboardInterrupt received, stopping node.')
    finally:
        # Ensure lifecycle flag is cleared and node cleaned up
        try:
            if node is not None:
                print('Destroying node and shutting down rclpy.')
                # This will set _active False via override
                node.destroy_node()
        except Exception:
            # ignore destroy exceptions
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
