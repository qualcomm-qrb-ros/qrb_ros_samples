# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import cv2
import os
from hand_palm_detector import nv12_to_bgr, resize_pad, denormalize_detections, BlazeDetector, BlazeLandmark
from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

# Import the custom TensorList message
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList

package_share_directory = get_package_share_directory("sample_hand_detection")

class HandLandmarkDetector(Node):
    def __init__(self):
        super().__init__('qrb_ros_hand_detector')

        # --- New publishers and subscribers for TensorList ---
        self.subscription = self.create_subscription(
            Image, 'image_raw', self.image_callback, 10
        )
        self.palm_detector_input_tensor_pub = self.create_publisher(
            TensorList, 'palm_detector_input_tensor', 10
        )
        self.landmark_detector_input_tensor_pub = self.create_publisher(
            TensorList, 'landmark_detector_input_tensor', 10
        )
        self.palm_detector_output_tensor_sub = self.create_subscription(
            TensorList, 'palm_detector_output_tensor', self.palm_result_callback, 10
        )
        self.landmark_detector_output_tensor_sub = self.create_subscription(
            TensorList, 'landmark_detector_output_tensor', self.landmark_result_callback, 10
        )
        self.publisher_ = self.create_publisher(
            Image, 'handlandmark_result', 10
        )
        # -----------------------------------------------------

        # --- New publishers and subscribers for TensorList ---
        self.bridge = CvBridge()
        self.latest_image = None
        self.affine2 = None
        self.box2 = None
        self.palm_detections = None

        self.processing = False
        self.waiting_for_palm_output = False
        self.waiting_for_landmark_output = False

        # Declare parameters model path
        self.declare_parameter('model_path', '/opt/model/')
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        # Check if the model path exists
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'MODEL_PATH does not exist: {self.model_path}')
            return

        # Initialize palm_detector and hand_regressor as instance variables
        self.palm_detector = BlazeDetector()
        self.palm_detector.load_anchors(os.path.join(self.model_path, "anchors_palm.npy"))
        self.hand_regressor = BlazeLandmark()

        # Create a timer to process images at a fixed rate
        #self.timer = self.create_timer(1.0, self.process_image)  # Process image every 1 second

    # --- New callback functions for TensorList messages ---
    def image_callback(self, msg):
        if self.processing:
            self.get_logger().info("Already processing an image, skipping this one.")
            return
        self.processing = True
        self.waiting_for_palm_output = True
        self.get_logger().info("Received image on image_raw topic")
        try:
            # Convert the ROS Image message to OpenCV format and process it
            if msg.encoding == "nv12":  # Check the encoding of the image
                nv12_data = np.frombuffer(msg.data, dtype=np.uint8)
                self.latest_image = nv12_to_bgr(nv12_data, msg.width, msg.height)  # Call the function to convert from NV12 to BGR
            elif msg.encoding == "bgr8":
                self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                self.get_logger().error(f"Unsupported image encoding: {msg.encoding}")
                return

            img1, img2, scale, pad = resize_pad(self.latest_image)
            img1 = self.palm_detector.palm_detector_qnn_preprocess(img1)

            # Convert the processed image to a TensorList message
            # Note: The image should be in the format expected by the model
            msg = TensorList()
            tensor = Tensor()
            tensor.data_type = 2
            tensor.name = "palm_detector_input_tensor"
            tensor.shape = [1, 256, 256, 3]
            tensor.data = img1.tobytes()
            msg.tensor_list.append(tensor)

            # Publish the TensorList message
            self.get_logger().info("Processed for palm detection, publishing TensorList")
            self.palm_detector_input_tensor_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error receiving image: {e}")

    def palm_result_callback(self, msg):
        if not self.waiting_for_palm_output:
            self.get_logger().info("Not waiting for palm output, skipping this callback.")
            return

        self.get_logger().info("Received TensorList on palm_detector_output_tensor")
        if len(msg.tensor_list) != 2:
            self.get_logger().error(f'Expected palm tensor_list size 2, got {len(msg.tensor_list)}')
            return
        try:
            img1, img2, scale, pad = resize_pad(self.latest_image)
            normalized_palm_detections = self.palm_detector.palm_detector_qnn_postprocess(msg)

            # Check if the output is a list with the expected shape
            if not (isinstance(normalized_palm_detections, list) and
                    len(normalized_palm_detections) == 1 and
                    hasattr(normalized_palm_detections[0], 'shape') and
                    normalized_palm_detections[0].shape == (1, 19)):
                # If the output is not valid, reset the flags and publish the original image
                self.get_logger().info("Palm detection shape is not valid, resetting and publishing original image.")
                self.processing = False
                self.waiting_for_palm_output = False
                self.waiting_for_landmark_output = False
                original_image = self.bridge.cv2_to_imgmsg(self.latest_image, encoding='bgr8')
                self.publisher_.publish(original_image)
                return
            # If the output is valid, proceed with processing
            self.waiting_for_palm_output = False
            self.waiting_for_landmark_output = True

            self.palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

            xc, yc, scale, theta = self.palm_detector.detection2roi(self.palm_detections)

            img, self.affine2, self.box2 = self.hand_regressor.extract_roi(self.latest_image, xc, yc, theta, scale)

            # Converts the image pixels to the range [-1, 1].
            img = img.astype(np.float32)
            msg = TensorList()
            tensor = Tensor()
            tensor.data_type = 2
            tensor.name = "landmark_detector_input_tensor"
            tensor.shape = [1, 256, 256, 3]
            tensor.data = img.tobytes()
            msg.tensor_list.append(tensor)

            # Publish the TensorList message
            self.get_logger().info("Received image and processed for palm detection, publishing TensorList")
            self.landmark_detector_input_tensor_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error processing palm detector output: {e}")

    def landmark_result_callback(self, msg):
        if not self.waiting_for_landmark_output:
            self.get_logger().info("Not waiting for landmark output, skipping this callback.")
            return
        self.waiting_for_landmark_output = False
        self.processing = False
        self.get_logger().info("Received TensorList on landmark_detector_output_tensor")
        # Add your processing logic here
        if len(msg.tensor_list) != 3:
            self.get_logger().error(f'Expected landmark tensor_list size 3, got {len(msg.tensor_list)}')
            return
        try:
            flags2, handed2, normalized_landmarks = self.hand_regressor.landmark_tensor_to_data(msg)

            landmarks = self.hand_regressor.denormalize_landmarks(normalized_landmarks, self.affine2)

            image = self.latest_image.copy()

            for i in range(len(flags2)):
                landmark, flag = landmarks[i], flags2[i]
                if flag > .5:
                    draw_landmarks(image, landmark[:, :2], HAND_CONNECTIONS, size=2)

            draw_roi(image, self.box2)
            draw_detections(image, self.palm_detections)

            processed_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')

            # Publish the  processed image
            self.get_logger().info("Received image and processed for landmark detection, publishing processed image")
            self.publisher_.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(f"Error processing palm detector output: {e}")

def main(args=None):
    rclpy.init(args=args)
    hand_detector_node = HandLandmarkDetector()
    try:
        rclpy.spin(hand_detector_node)
    except KeyboardInterrupt:
        print("HandLandmarkDetector node stopped by user")
    except Exception as e:
        print(f"Error in HandLandmarkDetector: {e}")
    finally:
        if hand_detector_node is not None:
            hand_detector_node.destroy_node()
        try:
            rclpy.shutdown()
        except rclpy._rclpy_pybind11.RCLError as e:
            print(f"rclpy already shutdown, not blocking: {e}")

if __name__ == '__main__':
    main()