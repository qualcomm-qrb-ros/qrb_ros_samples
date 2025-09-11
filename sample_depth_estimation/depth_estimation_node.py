# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import rclpy
import cv2
import math
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList


class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('qrb_ros_depth_estimator')
        self.publisher_ = self.create_publisher(Image, 'depth_map', 10)
        self.bridge = CvBridge()
        # Infer type
        self.origin_wh = tuple()
        self.scale = 1.0
        self.padding = tuple()

        self.get_logger().info("Use qnn inference node!")
        self.image_listener_ = self.create_subscription(Image, self.get_namespace() +'/cam0_stream1', self.image_callback, 10)           
        self.camera_listener_ = self.create_subscription(Image, '/image_raw', self.image_callback, 10)           

        self.qnn_infer_subscriber = self.create_subscription(
            TensorList,
            'qrb_inference_output_tensor',  # subscriber qnn infer output topic
            self.infer_callback,
            10
        )
        self.qnn_infer_publisher = self.create_publisher(
            TensorList,
            'qrb_inference_input_tensor',  # publish qnn infer input topic
            10
        )
        # Input tensor size
        self.target_size = [518, 518]
        self.msg_count = 0

    def nv12_to_bgr(self, nv12_image, width, height):
        """
        Convert NV12 image to BGR format.
        """
        yuv = nv12_image.reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        return bgr

    def image_callback(self, msg):
        if self.msg_count != 0:
            return
        if msg.encoding == 'nv12':
            nv12_data = np.frombuffer(msg.data, dtype=np.uint8)
            image = self.nv12_to_bgr(nv12_data, msg.width, msg.height)
        elif msg.encoding == 'bgr8':
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        else:
            self.get_logger().error(f'Unsupported image encoding: {msg.encoding}')
            return
        
        if image.shape[-1] != 3:
           raise ValueError("Need input 3 channels image")
        try:
            
            input, self.origin_wh, self.scale, self.padding = self.preprocess(image)
            msg = TensorList()
            tensor = Tensor()
            tensor.data_type = 2
            tensor.name = "depth_anything_input_tensor"
            tensor.shape = [1, 518, 518, 3]
            tensor.data = input.tobytes()
            msg.tensor_list.append(tensor)
                
            self.qnn_infer_publisher.publish(msg)
            self.msg_count += 1
            self.get_logger().info('Published qnn input tensor')
        except Exception as e:
          self.get_logger().error(f'Error preprocessing image: {e}')
        
    def infer_callback(self, msg):
        try:
            for result_tensor in msg.tensor_list:  # iterate tensor_list
                output_data = np.array(result_tensor.data)
                depth_map = output_data.view(np.float32)
                depth_color_map = self.postprocess(depth_map, self.origin_wh, self.scale, self.padding)                
                msg = self.bridge.cv2_to_imgmsg(depth_color_map, encoding='bgr8')
                self.publisher_.publish(msg)
                self.get_logger().info('Published depth map')
                self.msg_count = 0
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def preprocess(self, image):
        """
        Resize and pad image to be shape [..., dst_size[0], dst_size[1]]

        Parameters:
            image: (H, W, ...)
                Image to reshape.

        Returns:
            rescaled_padded_image: cv2.Image (..., dst_size[0], dst_size[1])
            scale: scale factor between original image and dst_size image, (w, h)
            pad: pixels of padding added to the rescaled image: (left_padding, top_padding)

        Based on https://github.com/zmurez/MediaPipePyTorch/blob/master/blazebase.py
        """
        height, width = image.shape[0:2]
        dst_frame_height, dst_frame_width = self.target_size[1], self.target_size[0]

        h_ratio = dst_frame_height / height
        w_ratio = dst_frame_width / width
        scale = min(h_ratio, w_ratio)
        if h_ratio < w_ratio:
            scale = h_ratio
            new_height = dst_frame_height
            new_width = math.floor(width * scale)
        else:
            scale = w_ratio
            new_height = math.floor(height * scale)
            new_width = dst_frame_width

        new_height = math.floor(height * scale)
        new_width = math.floor(width * scale)
        pad_h = dst_frame_height - new_height
        pad_w = dst_frame_width - new_width

        pad_top = int(pad_h // 2)
        pad_bottom = int(pad_h // 2 + pad_h % 2)
        pad_left = int(pad_w // 2)
        pad_right = int(pad_w // 2 + pad_w % 2)

        rescaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        rescaled_padded_image = cv2.copyMakeBorder(rescaled_image, pad_top, pad_bottom, pad_left, pad_right, 
                                cv2.BORDER_CONSTANT, value=(0,0,0))
        padding = (pad_left, pad_top)
        normalized = rescaled_padded_image.astype(np.float32) / 255.0
        return normalized[np.newaxis, ...], (width, height), scale, padding

    def postprocess(self, 
                    depth_map, 
                    origin_wh: tuple[int, int], 
                    scale: float, 
                    padding: tuple[int, int]
                    ):
        """Resize and undo padding on the depth map, then cenvert it to color map.

        Args:
            origin_wh (tuple[int, int]): origin image size
            scale (float): resize scale
            padding (tuple[int, int]): padding params(left, top)

        Returns:
            _type_: color map image
        """
        depth_map = depth_map.reshape((self.target_size[0], self.target_size[1], 1)) #Reshape to [518, 518, 1]

        output_image = self.undo_resize_pad(depth_map, origin_wh, scale, padding)
        # Normalize to [0,255]
        normalized = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
        return colored

    def undo_resize_pad(self,
            image: np.ndarray,
            orig_size_wh: tuple[int, int],
            scale: float,
            padding: tuple[int, int],
        ) -> np.ndarray:
        """
        Undoes the effect of resize_pad. Instead of scale, the original size
        (in order width, height) is provided to prevent an off-by-one size.
        """
        width, height = orig_size_wh
        # Rescale the image using OpenCV
        rescaled_image_np = cv2.resize(image, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
        # Calculate the scaled padding
        scaled_padding = [int(round(padding[0] / scale)), int(round(padding[1] / scale))]
        # Crop the image
        cropped_image_np = rescaled_image_np[
            scaled_padding[1] : scaled_padding[1] + height,
            scaled_padding[0] : scaled_padding[0] + width
        ]

        return cropped_image_np
    

def main(args=None):
    rclpy.init(args=args)
    depth_estimator_node = DepthEstimationNode()
    rclpy.spin(depth_estimator_node)
    depth_estimator_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 
