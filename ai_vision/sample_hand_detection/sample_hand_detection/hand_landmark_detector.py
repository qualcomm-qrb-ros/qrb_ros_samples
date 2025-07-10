# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import cv2
import os
from hand_palm_detector import resize_pad, denormalize_detections, BlazeDetector, BlazeLandmark
from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

package_share_directory = get_package_share_directory("sample_hand_detection")

class HandLandmarkDetector(Node):
    def __init__(self):
        super().__init__('qrb_ros_hand_detector')
        self.publisher_ = self.create_publisher(Image, 'handlandmark_result', 10)
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.latest_image = None
        self.processing = False

        # Declare parameters model path
        self.declare_parameter('model_path', '/opt/model/')
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.get_logger().info(f'MODEL_PATH set to: {self.model_path}')
        # Check if the model path exists
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'MODEL_PATH does not exist: {self.model_path}')
            return

        # Initialize palm_detector and hand_regressor as instance variables
        self.palm_detector = BlazeDetector()
        self.palm_detector.load_anchors(os.path.join(self.model_path, "anchors_palm.npy"))
        self.hand_regressor = BlazeLandmark()

        # Create a timer to process images at a fixed rate
        self.timer = self.create_timer(1.0, self.process_image)  # Process image every 1 second

    def hand_lankmark_processer(self, image):
        img1, img2, scale, pad = resize_pad(image)

        normalized_palm_detections = self.palm_detector.predict_on_image_np(img1, os.path.join(self.model_path, "MediaPipeHandDetector.tflite"))

        palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

        # Check the shape of palm_detections
        if len(palm_detections) == 0:
            self.get_logger().warn("No palm detections found.")
            return image  # Return the original image if no detections are found
        
        if np.array(palm_detections).shape[0] > 1:
            self.get_logger().warn("More than two palms are found, only support one palm detections")
            return image  # Return the original image if the palms shape is greater than 1
        
        xc, yc, scale, theta = self.palm_detector.detection2roi(palm_detections)

        img, affine2, box2 = self.hand_regressor.extract_roi(image, xc, yc, theta, scale)

        self.hand_regressor.qnn_preprocess(img)

        if not self.hand_regressor.run_network(os.path.join(self.model_path, "MediaPipeHandLandmarkDetector.tflite")):
            self.get_logger().warn("qtld-net-run is processing the model, please wait")
            return image
        
        # Post-process the results
        flags2, handed2, normalized_landmarks2 = self.hand_regressor.qnn_postprocess()
        landmarks2 = self.hand_regressor.denormalize_landmarks(normalized_landmarks2, affine2)

        for i in range(len(flags2)):
            landmark, flag = landmarks2[i], flags2[i]
            if flag > .5:
                draw_landmarks(image, landmark[:, :2], HAND_CONNECTIONS, size=2)

        draw_roi(image, box2)
        draw_detections(image, palm_detections)

        return image

    def image_callback(self, msg):
        try:
            # Store the latest image
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error receiving image: {e}")

    def process_image(self):
        if self.latest_image is not None and not self.processing:
            self.processing = True
            try:
                # Process the latest image
                processed_frame = self.hand_lankmark_processer(self.latest_image)
                # Convert the processed image back to a ROS Image message
                processed_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding='bgr8')
                # Publish the processed image
                self.publisher_.publish(processed_msg)
                self.get_logger().info('Publishing image')
            except Exception as e:
                self.get_logger().error(f"Error processing image: {e}")
            finally:
                self.processing = False

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
        if rclpy.ok():
            hand_detector_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
