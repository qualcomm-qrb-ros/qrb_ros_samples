# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import rclpy
import cv2
import os
import time
import threading
import uuid
import numpy as np
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, ReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sample_face_detection.mediapipe_face_base import resize_pad, denormalize_detections, BlazeDetector, BlazeLandmark
from sample_face_detection.draw import draw_detections, draw_landmarks, draw_roi, FACE_CONNECTIONS

# Import the custom TensorList message
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList

# Timeout for waiting for inference outputs (seconds)
DEFAULT_TIMEOUT = 1.0
# Timeout for landmark detector (seconds)
LANDMARK_TIMEOUT = 2.0

class MediaFaceDetNode(Node):
    def __init__(self):
        super().__init__('mediaface_det_node')

        # QoS profiles
        # 1) TensorList topics: RELIABLE + KEEP_LAST depth=1 (only keep latest frame)
        self.qos_tensor = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE
        )
        # 2) Image topics: depth=1 to avoid frame accumulation
        self.qos_image = QoSProfile(depth=1)

        # Create publisher and subscribers with QoS
        self.subscription = self.create_subscription(
            Image, 'image_raw', self.image_callback_image, qos_profile=self.qos_image
        )
        self.face_detector_input_tensor_pub = self.create_publisher(
            TensorList, 'face_detector_input_tensor', qos_profile=self.qos_tensor
        )
        self.face_landmark_input_tensor_pub = self.create_publisher(
            TensorList, 'face_landmark_input_tensor', qos_profile=self.qos_tensor
        )
        self.face_detector_output_tensor_sub = self.create_subscription(
            TensorList, 'face_detector_output_tensor', self.facedet_result_callback, qos_profile=self.qos_tensor
        )
        self.face_landmark_output_tensor_sub = self.create_subscription(
            TensorList, 'face_landmark_output_tensor', self.landmark_result_callback, qos_profile=self.qos_tensor
        )
        self.publisher = self.create_publisher(
            Image, 'mediaface_det_image', qos_profile=self.qos_image
        )

        # Initialize variables
        self.latest_image = None
        self.bridge = CvBridge()
        self.affine2 = None
        self.box2 = None
        self.face_detections = None
        
        # Synchronization and state
        self.processing = False
        self.waiting_for_face_output = False
        self.waiting_for_landmark_output = False
        self.lock = threading.Lock()
        self.last_request_time = 0.0  # timestamp of last face request
        
        # Per-request id for correlating publish/receive across nodes
        self.current_request_id = None
        
        # When a timeout occurs, we set paused_until to a timestamp in the future.
        # While time.time() < paused_until the node will ignore incoming image messages.
        self.paused_until = 0.0
        
        # Shutdown flag to prevent callbacks from executing during node destruction
        self.is_shutting_down = False

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
        
        # Start a timer to check for timeouts (1 Hz)
        self.create_timer(1.0, self._timeout_check)

    def _timeout_check(self):
        """Check if we have been waiting too long for outputs and reset state if needed."""
        now = time.time()
        with self.lock:
            # If we're currently paused, check if pause has expired and clear it.
            if self.paused_until and now >= self.paused_until:
                self.get_logger().warn("Pause after timeout expired, resuming message handling.")
                self.paused_until = 0.0
                # Ensure processing flags are cleared
                self.processing = False
                self.waiting_for_face_output = False
                self.waiting_for_landmark_output = False

            # Timeout waiting for face output - 1 second timeout
            if self.processing and self.waiting_for_face_output:
                elapsed = now - self.last_request_time
                if elapsed > DEFAULT_TIMEOUT:
                    try:
                        self.get_logger().warn(
                            f"⚠️ Face detector timeout ({elapsed:.2f}s > {DEFAULT_TIMEOUT}s) "
                            f"- Skipping frame {self.current_request_id}, ready for next frame"
                        )
                    except Exception:
                        pass
                    # Reset state immediately, no pause - ready to process next frame
                    self.processing = False
                    self.waiting_for_face_output = False
                    self.waiting_for_landmark_output = False
                    self.current_request_id = None
                    return

            # Timeout waiting for landmark output - 2 second timeout
            if self.processing and self.waiting_for_landmark_output:
                elapsed = now - self.last_request_time
                if elapsed > LANDMARK_TIMEOUT:
                    try:
                        self.get_logger().warn(
                            f"⚠️ Landmark detector timeout ({elapsed:.2f}s > {LANDMARK_TIMEOUT}s) "
                            f"- Skipping frame {self.current_request_id}, ready for next frame"
                        )
                    except Exception:
                        pass
                    # Reset state immediately, no pause - ready to process next frame
                    self.processing = False
                    self.waiting_for_face_output = False
                    self.waiting_for_landmark_output = False
                    self.current_request_id = None
                    return

    def image_callback_image(self, msg):
        if self.is_shutting_down:
            return
            
        now = time.time()
        with self.lock:
            # If currently paused due to a previous timeout, skip receiving messages
            if self.paused_until and now < self.paused_until:
                # Reduced verbosity: skip noisy warn per-frame while paused
                try:
                    self.get_logger().debug(f"Ignoring incoming image because node is paused until {self.paused_until:.3f} (now={now:.3f}).")
                except Exception:
                    pass
                return

            if self.processing or self.waiting_for_face_output or self.waiting_for_landmark_output:
                try:
                    self.get_logger().debug("Already processing an image, skipping this one.")
                except Exception:
                    pass
                return
            
            self.processing = True
            self.last_request_time = now
            self.current_request_id = str(uuid.uuid4())[:8]

        try:
            if msg.encoding == 'nv12':
                nv12_img = np.frombuffer(msg.data, dtype=np.uint8)
                height = msg.height
                width = msg.width
                nv12_img = nv12_img.reshape((height + height//2, width))
                self.latest_image = cv2.cvtColor(nv12_img, cv2.COLOR_YUV2BGR_NV12)
            else:
                self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            img1, img2, scale, pad = resize_pad(self.latest_image)
            img1 = self.face_detector.face_detector_qnn_preprocess(img1)

            tensor_msg = TensorList()
            tensor = Tensor()
            tensor.data_type = 2
            tensor.name = f"face_detector_input_tensor_{self.current_request_id}"
            tensor.shape = [1, 256, 256, 3]
            tensor.data = img1.tobytes()
            tensor_msg.tensor_list.append(tensor)

            with self.lock:
                self.waiting_for_face_output = True
                self.last_request_time = time.time()
            
            if not self.is_shutting_down:
                self.face_detector_input_tensor_pub.publish(tensor_msg)
        except Exception:
            with self.lock:
                self.processing = False
                self.waiting_for_face_output = False


    def facedet_result_callback(self, msg):
        if self.is_shutting_down:
            return

        now = time.time()
        # NOTE: do NOT ignore inference outputs because of paused_until.
        # Log when we received the TensorList back from inference
        try:
            self.get_logger().info(f"[recv] face_result_callback called at {now:.6f}, waiting_for_face_output={self.waiting_for_face_output}, current_request_id={self.current_request_id}")
        except Exception:
            return  # If logging fails, context is invalid, exit early

        with self.lock:
            if not self.waiting_for_face_output:
                return
            self.waiting_for_face_output = False
        
        if len(msg.tensor_list) != 2:
            with self.lock:
                self.processing = False
                self.current_request_id = None
            return

        try:
            img1, img2, scale, pad = resize_pad(self.latest_image)
            normalized_face_detections = self.face_detector.face_detector_qnn_postprocess(msg)

            # Check if detection is valid
            if (normalized_face_detections is None or 
                normalized_face_detections.shape[0] == 0 or
                len(normalized_face_detections) != 1 or
                normalized_face_detections[0].shape != (17,)):
                with self.lock:
                    self.processing = False
                    self.current_request_id = None
                return

            self.face_detections = denormalize_detections(normalized_face_detections, scale, pad)
            xc, yc, scale, theta = self.face_detector.detection2roi(self.face_detections)
            img, self.affine2, self.box2 = self.face_regressor.extract_roi(self.latest_image, xc, yc, theta, scale)

            img = img.astype(np.float32)
            tensor_msg = TensorList()
            tensor = Tensor()
            tensor.data_type = 2
            tensor.name = f"landmark_detector_input_tensor_{self.current_request_id}"
            tensor.shape = [1, 192, 192, 3]
            tensor.data = img.tobytes()
            tensor_msg.tensor_list.append(tensor)
            
            with self.lock:
                self.last_request_time = time.time()
                self.waiting_for_landmark_output = True
            
            if not self.is_shutting_down:
                self.face_landmark_input_tensor_pub.publish(tensor_msg)

        except Exception:
            with self.lock:
                self.processing = False
                self.waiting_for_face_output = False
                self.waiting_for_landmark_output = False
                self.current_request_id = None

    def landmark_result_callback(self, msg):
        if self.is_shutting_down:
            return
        now = time.time()
        
        # NOTE: do NOT ignore inference outputs because of paused_until.
        try:
            self.get_logger().info(f"[recv] landmark_result_callback called at {now:.6f}, current_request_id={self.current_request_id}, waiting_for_landmark_output={self.waiting_for_landmark_output}")
        except Exception:
            return  # If logging fails, context is invalid, exit early

        with self.lock:
            if not self.waiting_for_landmark_output:
                return
            self.waiting_for_landmark_output = False

        if len(msg.tensor_list) != 2:
            with self.lock:
                self.processing = False
                self.current_request_id = None
            return

        try:
            flags2, normalized_landmarks = self.face_regressor.landmark_tensor_to_data(msg)

            if normalized_landmarks is None or normalized_landmarks.shape[0] == 0:
                with self.lock:
                    self.processing = False
                return

            landmarks = self.face_regressor.denormalize_landmarks(normalized_landmarks, self.affine2)
            image = self.latest_image.copy()

            for i in range(len(flags2)):
                landmark, flag = landmarks[i], flags2[i]
                if flag > .3:
                    draw_landmarks(image, landmark[:, :2], FACE_CONNECTIONS, size=2)
            
            draw_roi(image, self.box2)
            draw_detections(image, self.face_detections)
            processed_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')

            with self.lock:
                self.processing = False
            
            if not self.is_shutting_down:
                self.publisher.publish(processed_msg)

            with self.lock:
                self.current_request_id = None

        except Exception as e:
            with self.lock:
                self.processing = False
                self.waiting_for_face_output = False
                self.waiting_for_landmark_output = False
                self.current_request_id = None

def main(args=None):
    rclpy.init(args=args)
    face_detector_node = None
    executor = None

    try:
        face_detector_node = MediaFaceDetNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(face_detector_node)
        executor.spin()
    except KeyboardInterrupt:
        print("FaceLandmarkDetector node stopped by user")
        if face_detector_node is not None:
            face_detector_node.is_shutting_down = True
        time.sleep(0.1)
    except Exception as e:
        print(f"Error in FaceLandmarkDetector: {e}")

    finally:
        if face_detector_node is not None:
            face_detector_node.is_shutting_down = True
            try:
                face_detector_node.destroy_node()
            except Exception as e:
                print(f"Node destruction error (non-fatal): {e}")

        if executor is not None:
            try:
                executor.shutdown(timeout_sec=2.0)
            except Exception as e:
                print(f"Executor shutdown error (non-fatal): {e}")

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as e:
                print(f"rclpy shutdown error (non-fatal): {e}")

if __name__ == '__main__':
    main()
