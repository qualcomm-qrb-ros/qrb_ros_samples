# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import numpy as np
import cv2
import rclpy
from sample_hrnet_pose_estimation import preprocess, postprocess
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList
from ament_index_python.packages import get_package_share_directory

package_share_directory = get_package_share_directory("sample_hrnet_pose_estimation")

# raw image subscriber -> preprocess -> nn_inference_input_tensor publisher -> nn_inference_output_tensor subscriber -> postprocess -> pose_estimation_results&pose_estimation_points publisher

class hrnet_pose_estimation(Node):
    def __init__(self):
        super().__init__('hrnet_pose_estimation_node')

        self.raw_image_subscription = self.create_subscription(Image, 'image_raw', self.raw_image_callback, 10)
        self.pose_detection_result_image_publisher = self.create_publisher(Image, 'pose_estimation_results', 10)
        self.pose_detection_result_points_publisher = self.create_publisher(PolygonStamped, 'pose_estimation_points', 10)
        self.nn_inference_output_tensor_subscription = self.create_subscription(TensorList, 'qrb_inference_output_tensor', self.nn_inference_callback, 10)
        self.nn_inference_input_tensor_publisher = self.create_publisher(TensorList, 'qrb_inference_input_tensor', 10)

        self.raw_image_processed_flag = True
        self.preprocessed_image = None

        self.bridge = CvBridge()

    def nv12_to_bgr(self, nv12_image, width, height):
        """
        Convert NV12 image to BGR format.
        """
        yuv = nv12_image.reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        return bgr

    # define call back function
    def raw_image_callback(self, msg):

        self.get_logger().info('Received raw_image message')
        if msg.encoding == 'nv12':
            nv12_data = np.frombuffer(msg.data, dtype=np.uint8)
            cv_image = self.nv12_to_bgr(nv12_data, msg.width, msg.height)
        elif msg.encoding == 'bgr8':
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        else:
            self.get_logger().error(f'Unsupported image encoding: {msg.encoding}')
            return

        preprocessed_image = preprocess.preprocess(cv_image)

        if self.raw_image_processed_flag == True:
            self.cv_image = cv_image
            nn_inference_input_tensor_msg = self.make_nn_inference_input_tensor_msg(msg, preprocessed_image)
            self.nn_inference_input_tensor_publisher.publish(nn_inference_input_tensor_msg)
            self.get_logger().info("Publish nn_inference_input_tensor_msg")
            self.raw_image_processed_flag = False
        else:
            self.get_logger().info("Skip nn_inference_input_tensor_msg")

    def make_nn_inference_input_tensor_msg(self, original_msg, preprocessed_image):
        nn_inference_input_tensor_msg = TensorList()
        nn_inference_input_tensor_msg.header = original_msg.header

        tensor = Tensor()
        tensor.name = "pose detection nn_inference_input_tensor"
        tensor.data_type = 2    #float32
        tensor.shape = preprocessed_image.shape
        tensor.data = preprocessed_image.tobytes()

        nn_inference_input_tensor_msg.tensor_list.append(tensor)
        return nn_inference_input_tensor_msg

    def nn_inference_callback(self, nn_inference_output_tensor_msg):
        self.get_logger().info('Received nn_inference_output_tensor message')

        for result in nn_inference_output_tensor_msg.tensor_list:
            output_np_array = np.array(result.data)
            output_np_array = output_np_array.view(np.float32)
            result_image,result_points = postprocess.postprocess(self.cv_image,output_np_array)
            result_points_msg = self.make_pose_estimation_points_msg(result_points, nn_inference_output_tensor_msg)
            self.publish_result_image_message(result_image,result_points_msg)
        self.raw_image_processed_flag = True

    def make_pose_estimation_points_msg(self, points, nn_inference_output_tensor_msg):
        result_points_msg = PolygonStamped()
        result_points_msg.header = nn_inference_output_tensor_msg.header
        for point in points:
            point_msg = Point32()
            point_msg.x = float(point[0])
            point_msg.y = float(point[1])
            result_points_msg.polygon.points.append(point_msg)
        return result_points_msg

    def publish_result_image_message(self, image, result_points_msg):
        ros_image = self.bridge.cv2_to_imgmsg(image,encoding="bgr8")
        self.pose_detection_result_image_publisher.publish(ros_image)
        self.pose_detection_result_points_publisher.publish(result_points_msg)
        self.get_logger().info('Publisher pose_estimation_results&pose_estimation_points message')


def main(args=None):
    rclpy.init(args=args)
    node = hrnet_pose_estimation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
