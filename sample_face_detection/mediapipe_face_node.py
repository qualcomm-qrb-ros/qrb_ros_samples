# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import rclpy
import cv2
import os
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sample_face_detection.mediapipe_face_base import resize_pad, denormalize_detections, BlazeDetector, BlazeLandmark
from sample_face_detection.draw import draw_detections, draw_landmarks, draw_roi, FACE_CONNECTIONS

# Import the custom TensorList message
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList

class MediaFaceDetNode(Node):
    def __init__(self):
        super().__init__('mediaface_det_node')

        # Create publisher and subscribers
        self.subscription = self.create_subscription(Image, 'image_raw', self.image_callback_image,10)
        self.face_detector_input_tensor_pub = self.create_publisher(TensorList, 'face_detector_input_tensor', 10)
        self.face_landmark_input_tensor_pub = self.create_publisher(TensorList, 'face_landmark_input_tensor', 10)
        self.face_detector_output_tensor_sub = self.create_subscription(TensorList, 'face_detector_output_tensor', self.facedet_result_callback, 10)
        self.face_landmark_output_tensor_sub = self.create_subscription(TensorList, 'face_landmark_output_tensor', self.landmark_result_callback, 10)
        self.publisher = self.create_publisher(Image, 'mediaface_det_image', 10)

        # Initialize variables
        self.latest_image = None
        self.bridge = CvBridge()
        self.affine2 = None
        self.box2 = None
        self.face_detections = None
        self.processing = False
        self.waiting_for_face_output = False
        self.waiting_for_landmark_output = False

        # Declare parameters model path
        self.declare_parameter('model_path', '/opt/model/')
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.get_logger().info(f'MODEL_PATH set to: {self.model_path}')
        # Check if the model path exists
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'MODEL_PATH does not exist: {self.model_path}')
            return

        # Initialize face_detector and face_regressor as instance variables
        self.face_detector = BlazeDetector()
        self.face_detector.load_anchors(os.path.join(self.model_path, "anchors_face.npy"))
        self.face_regressor = BlazeLandmark()
        self.get_logger().info('init done~')

    def image_callback_image(self, msg):
        if self.processing:
            self.get_logger().info("Already processing an image, skipping this one.")
            return
        self.processing = True
        self.waiting_for_face_output = True
        self.get_logger().info("Received image on image_raw topic")

        try:
            if msg.encoding == 'nv12':
                # Convert YUV12 to RGB
                nv12_img = np.frombuffer(msg.data, dtype=np.uint8)
                height = msg.height
                width = msg.width
                nv12_img = nv12_img.reshape((height + height//2, width))
                # convert the NV12 image to BGR
                self.latest_image = cv2.cvtColor(nv12_img, cv2.COLOR_YUV2BGR_NV12)
            else:
                # Assume other formats are already in RGB or can be directly converted
                self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process the input image
            img1, img2, scale, pad = resize_pad(self.latest_image)
            img1 = self.face_detector.face_detector_qnn_preprocess(img1)

            # Convert the processed image to a TensorList message
            msg = TensorList()
            tensor = Tensor()
            tensor.data_type = 2
            tensor.name = "face_detector_input_tensor"
            tensor.shape = [1, 256, 256, 3]
            tensor.data = img1.tobytes()
            msg.tensor_list.append(tensor)

            # Publish the TensorList message
            self.get_logger().info("Processed for face detection, publishing TensorList")
            self.face_detector_input_tensor_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error receiving image: {e}")


    def facedet_result_callback(self, msg):
        if not self.waiting_for_face_output:
            self.get_logger().info("Not waiting for face output, skipping this callback.")
            return
        self.get_logger().info("Received TensorList on face_detector_output_tensor")
        if len(msg.tensor_list) != 2:
            self.get_logger().error(f'Expected face tensor_list size 2, got {len(msg.tensor_list)}')
            return

        try:
            # Post process face detector model output
            img1, img2, scale, pad = resize_pad(self.latest_image)
            normalized_face_detections = self.face_detector.face_detector_qnn_postprocess(msg)

            if normalized_face_detections is None or normalized_face_detections.shape[0] == 0:
                self.get_logger().info('No face detect is found, skipping this frame.')
                self.processing = False
                self.waiting_for_face_output = False
                self.waiting_for_landmark_output = False
                return

            # Check if the output is a list with the expected shape
            if not ( len(normalized_face_detections) == 1 and
                    normalized_face_detections[0].shape == (17,)):
                # If the output is not valid, reset the flags and publish the original image
                self.get_logger().info("Face detection shape is not valid, resetting and publishing original image.")
                self.processing = False
                self.waiting_for_face_output = False
                self.waiting_for_landmark_output = False
                # Note: original code had self.publisher_., changed to self.publisher
                original_image = self.bridge.cv2_to_imgmsg(self.latest_image, encoding='bgr8')
                self.publisher.publish(original_image)
                return

            # If the output is valid, proceed with processing
            self.waiting_for_face_output = False
            self.waiting_for_landmark_output = True
            self.face_detections = denormalize_detections(normalized_face_detections, scale, pad)
            xc, yc, scale, theta = self.face_detector.detection2roi(self.face_detections)
            img, self.affine2, self.box2 = self.face_regressor.extract_roi(self.latest_image, xc, yc, theta, scale)

            # Converts the image pixels to the range [-1, 1].
            img = img.astype(np.float32)
            msg = TensorList()
            tensor = Tensor()
            tensor.data_type = 2
            tensor.name = "landmark_detector_input_tensor"
            tensor.shape = [1, 192, 192, 3]
            tensor.data = img.tobytes()
            msg.tensor_list.append(tensor)

            # Publish the TensorList message
            self.get_logger().info("Received image and processed for face detection, publishing TensorList")
            self.face_landmark_input_tensor_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error processing face detector output: {e}")

    def landmark_result_callback(self, msg):
        if not self.waiting_for_landmark_output:
            self.get_logger().info("Not waiting for landmark output, skipping this callback.")
            return
        self.waiting_for_landmark_output = False
        self.processing = False
        self.get_logger().info("Received TensorList on landmark_detector_output_tensor")

        if len(msg.tensor_list) != 2:
            self.get_logger().error(f'Expected landmark tensor_list size 2, got {len(msg.tensor_list)}')
            return

        try:
            flags2, normalized_landmarks = self.face_regressor.landmark_tensor_to_data(msg)

            if normalized_landmarks is None or normalized_landmarks.shape[0] == 0:
                self.get_logger().info('No face landmark is found, skipping this frame.')
                return

            landmarks = self.face_regressor.denormalize_landmarks(normalized_landmarks, self.affine2)
            image = self.latest_image.copy()

            for i in range(len(flags2)):
                landmark, flag = landmarks[i], flags2[i]
                if flag > .3:
                    draw_landmarks(image, landmark[:, :2], FACE_CONNECTIONS, size=2)
            
            # draw face detection & face landmark point
            draw_roi(image, self.box2)
            draw_detections(image, self.face_detections)
            processed_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')

            # Publish the  processed image
            self.get_logger().info("Received image and processed for landmark detection, publishing processed image")
            self.publisher.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(f"Error processing face landmark output: {e}")

def main(args=None):
    rclpy.init(args=args)
    face_detector_node = MediaFaceDetNode()
    try:
        rclpy.spin(face_detector_node)
    except KeyboardInterrupt:
        print("FaceLandmarkDetector node stopped by user")
    except Exception as e:
        print(f"Error in FaceLandmarkDetector: {e}")
    finally:
        if face_detector_node is not None:
            face_detector_node.destroy_node()
        try:
            rclpy.shutdown()
        except rclpy._rclpy_pybind11.RCLError as e:
            print(f"rclpy already shutdown, not blocking: {e}")

if __name__ == '__main__':
    main()
