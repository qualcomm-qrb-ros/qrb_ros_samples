# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import cv2
import os
import time
import threading
import uuid
from hand_palm_detector import nv12_to_bgr, resize_pad, denormalize_detections, BlazeDetector, BlazeLandmark
from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, ReliabilityPolicy
from rclpy.logging import LoggingSeverity, set_logger_level
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

# Import the custom TensorList message
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList

package_share_directory = get_package_share_directory("sample_hand_detection")

# Timeout for waiting for inference outputs (seconds)
DEFAULT_TIMEOUT = 3.0
# Pause duration after a timeout (seconds)
TIMEOUT_PAUSE_SECONDS = 6.0

class HandLandmarkDetector(Node):
    def __init__(self):
        super().__init__('qrb_ros_hand_detector')

        # Reduce log verbosity: set node logger to WARN by default
        try:
            set_logger_level(self.get_name(), LoggingSeverity.INFO)
        except Exception:
            # Fallback in case of logging API differences
            pass

        # QoS profiles
        # 1) TensorList topics: RELIABLE + KEEP_LAST depth=10
        self.qos_tensor = QoSProfile(
            depth=20,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE
        )
        # 2) Image topics: increase history depth to 20 (default RELIABLE/KEEP_LAST)
        self.qos_image = QoSProfile(depth=20)

        # --- Publishers and subscribers with QoS ---
        self.subscription = self.create_subscription(
            Image, 'image_raw', self.image_callback, qos_profile=self.qos_image
        )
        self.palm_detector_input_tensor_pub = self.create_publisher(
            TensorList, 'palm_detector_input_tensor', qos_profile=self.qos_tensor
        )
        self.landmark_detector_input_tensor_pub = self.create_publisher(
            TensorList, 'landmark_detector_input_tensor', qos_profile=self.qos_tensor
        )
        self.palm_detector_output_tensor_sub = self.create_subscription(
            TensorList, 'palm_detector_output_tensor', self.palm_result_callback, qos_profile=self.qos_tensor
        )
        self.landmark_detector_output_tensor_sub = self.create_subscription(
            TensorList, 'landmark_detector_output_tensor', self.landmark_result_callback, qos_profile=self.qos_tensor
        )
        self.publisher_ = self.create_publisher(
            Image, 'handlandmark_result', qos_profile=self.qos_image
        )
        # -----------------------------------------------------

        self.bridge = CvBridge()
        self.latest_image = None
        self.affine2 = None
        self.box2 = None
        self.palm_detections = None

        # Synchronization and state
        self.processing = False
        self.waiting_for_palm_output = False
        self.waiting_for_landmark_output = False
        self.lock = threading.Lock()
        self.last_request_time = 0.0  # timestamp of last palm request

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
        # Check if the model path exists
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'MODEL_PATH does not exist: {self.model_path}')
            return

        # Initialize palm_detector and hand_regressor as instance variables
        self.palm_detector = BlazeDetector()
        self.palm_detector.load_anchors(os.path.join(self.model_path, "anchors_palm.npy"))
        self.hand_regressor = BlazeLandmark()

        # Start a timer to check for timeouts (1 Hz)
        self.create_timer(1.0, self._timeout_check)

    def _timeout_check(self):
        """Check if we have been waiting too long for outputs and reset state if needed.

        On timeout we will pause processing (ignore new incoming images) for
        TIMEOUT_PAUSE_SECONDS and reset internal state. The paused_until timestamp
        is used to gate incoming images in image_callback.
        """
        now = time.time()
        with self.lock:
            # If we're currently paused, check if pause has expired and clear it.
            if self.paused_until and now >= self.paused_until:
                self.get_logger().warn("Pause after timeout expired, resuming message handling.")
                self.paused_until = 0.0
                # Ensure processing flags are cleared
                self.processing = False
                self.waiting_for_palm_output = False
                self.waiting_for_landmark_output = False

            # Timeout waiting for palm output
            if self.processing and self.waiting_for_palm_output:
                elapsed = now - self.last_request_time
                if elapsed > DEFAULT_TIMEOUT:
                    self.get_logger().warn(f"Timeout waiting for palm output ({elapsed:.2f}s). Pausing for {TIMEOUT_PAUSE_SECONDS:.1f}s and resetting state.")
                    self.paused_until = now + TIMEOUT_PAUSE_SECONDS
                    self.last_request_time = now
                    self.processing = False
                    self.waiting_for_palm_output = False
                    self.waiting_for_landmark_output = False
                    self.current_request_id = None
                    return

            # Timeout waiting for landmark output (use a longer threshold)
            if self.processing and self.waiting_for_landmark_output:
                elapsed = now - self.last_request_time
                if elapsed > DEFAULT_TIMEOUT * 2:
                    self.get_logger().warn(f"Timeout waiting for landmark output ({elapsed:.2f}s). Pausing for {TIMEOUT_PAUSE_SECONDS:.1f}s and resetting state.")
                    self.paused_until = now + TIMEOUT_PAUSE_SECONDS
                    self.last_request_time = now
                    self.processing = False
                    self.waiting_for_palm_output = False
                    self.waiting_for_landmark_output = False
                    self.current_request_id = None
                    return

    # --- New callback functions for TensorList messages ---
    def image_callback(self, msg):
        # Check shutdown flag first to avoid processing during shutdown
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

            if self.processing or self.waiting_for_palm_output or self.waiting_for_landmark_output:
                try:
                    self.get_logger().debug("Already processing an image, skipping this one.")
                except Exception:
                    pass
                return
            # Mark as processing and waiting for palm output
            self.processing = True
            self.last_request_time = now
            # generate a per-request id for correlation and debugging
            self.current_request_id = str(uuid.uuid4())[:8]

        try:
            self.get_logger().info(f"[{self.current_request_id}] Received image on image_raw topic at {now:.6f}")
        except Exception:
            pass

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
            tensor_msg = TensorList()
            tensor = Tensor()
            tensor.data_type = 2
            # include request id and timestamp in tensor name so inference node logs (if any) can be correlated
            tensor.name = f"palm_detector_input_tensor_{self.current_request_id}"
            tensor.shape = [1, 256, 256, 3]
            tensor.data = img1.tobytes()
            tensor_msg.tensor_list.append(tensor)

            # Publish the TensorList message
            publish_time = time.time()
            try:
                self.get_logger().debug(f"[{self.current_request_id}] PreProcessed for palm detection, publishing TensorList at {publish_time:.6f}")
            except Exception:
                pass
            with self.lock:
                self.waiting_for_palm_output = True
                self.last_request_time = publish_time
            
            # Check shutdown flag before publishing
            if not self.is_shutting_down:
                self.palm_detector_input_tensor_pub.publish(tensor_msg)
                try:
                    self.get_logger().info(f"[{self.current_request_id}] PreProcessed for palm detection, published TensorList at {time.time():.6f}")
                except Exception:
                    # Context may be invalid during shutdown, ignore logging errors
                    pass
        except Exception as e:
            try:
                self.get_logger().error(f"Error receiving image: {e}")
            except Exception:
                pass

    def palm_result_callback(self, msg):
        # Check shutdown flag first to avoid using destroyed resources
        if self.is_shutting_down:
            return

        now = time.time()
        # NOTE: do NOT ignore inference outputs because of paused_until.
        # Log when we received the TensorList back from inference
        try:
            self.get_logger().info(f"[recv] palm_result_callback called at {now:.6f}, waiting_for_palm_output={self.waiting_for_palm_output}, current_request_id={self.current_request_id}")
        except Exception:
            return  # If logging fails, context is invalid, exit early

        with self.lock:
            if not self.waiting_for_palm_output:
                try:
                    self.get_logger().debug("Not waiting for palm output, skipping this callback.")
                except Exception:
                    pass
                return
            # Clear the waiting flag (we will set landmark waiting flag later if proceed)
            self.waiting_for_palm_output = False

        try:
            self.get_logger().debug("Received TensorList on palm_detector_output_tensor")
        except Exception:
            pass

        # Defensive check: if the TensorList structure is not as expected, log details and reset state
        if len(msg.tensor_list) != 2:
            try:
                self.get_logger().error(f'Expected palm tensor_list size 2, got {len(msg.tensor_list)}')
                # log details for debugging
                for i, t in enumerate(msg.tensor_list):
                    try:
                        self.get_logger().error(f" Palm Tensor[{i}]: name={t.name}, dtype={t.data_type}, shape={t.shape}, data_len={len(t.data) if t.data else 0}")
                    except Exception:
                        self.get_logger().error(f" Palm Tensor[{i}]: unable to read meta")
            except Exception:
                pass
            # reset processing state
            with self.lock:
                self.processing = False
                self.current_request_id = None
            return

        try:
            img1, img2, scale, pad = resize_pad(self.latest_image)
            normalized_palm_detections = self.palm_detector.palm_detector_qnn_postprocess(msg)

            # Check if the output is a list with the expected shape
            is_valid = (isinstance(normalized_palm_detections, list) and
                        len(normalized_palm_detections) == 1 and
                        hasattr(normalized_palm_detections[0], 'shape') and
                        normalized_palm_detections[0].shape == (1, 19))

            # Log the shape/type returned by postprocess for debugging, with request id
            try:
                self.get_logger().debug(f"[{self.current_request_id}] normalized_palm_detections type={type(normalized_palm_detections)}")
                if isinstance(normalized_palm_detections, list):
                    for i, arr in enumerate(normalized_palm_detections):
                        self.get_logger().debug(f"[{self.current_request_id}]  item[{i}] shape/len: {getattr(arr,'shape',None)}")
            except Exception:
                pass

            if not is_valid:
                # Log details to help debugging: print tensor meta & postprocess result summary
                try:
                    self.get_logger().debug("Palm detection shape is not valid, dumping tensor meta and resetting.")
                    for i, t in enumerate(msg.tensor_list):
                        try:
                            self.get_logger().debug(f" Palm Tensor[{i}]: name={t.name}, dtype={t.data_type}, shape={t.shape}, data_len={len(t.data) if t.data else 0}")
                        except Exception:
                            self.get_logger().debug(f" Palm Tensor[{i}]: unable to read meta")

                    # Try to log what postprocess returned (type/shape)
                    self.get_logger().debug(f"[{self.current_request_id}] normalized_palm_detections type={type(normalized_palm_detections)}")
                    if isinstance(normalized_palm_detections, list):
                        for i, arr in enumerate(normalized_palm_detections):
                            try:
                                self.get_logger().debug(f"[{self.current_request_id}]  item[{i}] shape/len: {getattr(arr,'shape',None)}")
                            except Exception:
                                pass
                except Exception:
                    pass

                # Check shutdown before publishing
                if not self.is_shutting_down:
                    try:
                        original_image = self.bridge.cv2_to_imgmsg(self.latest_image, encoding='bgr8')
                        self.publisher_.publish(original_image)
                        self.get_logger().info("Published original image due to invalid palm detection output.")
                    except Exception:
                        pass

                with self.lock:
                    self.processing = False
                    self.current_request_id = None

                return

            self.palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

            xc, yc, scale, theta = self.palm_detector.detection2roi(self.palm_detections)

            img, self.affine2, self.box2 = self.hand_regressor.extract_roi(self.latest_image, xc, yc, theta, scale)

            # Converts the image pixels to the range [-1, 1].
            img = img.astype(np.float32)
            tensor_msg = TensorList()
            tensor = Tensor()
            tensor.data_type = 2
            # Include same request id in landmark input so inference node can log it back
            tensor.name = f"landmark_detector_input_tensor_{self.current_request_id}"
            tensor.shape = [1, 256, 256, 3]
            tensor.data = img.tobytes()
            tensor_msg.tensor_list.append(tensor)

            try:
                self.get_logger().debug(f"[{self.current_request_id}] Publishing TensorList for landmark at {time.time():.6f}")
            except Exception:
                pass
            
            with self.lock:
                self.last_request_time = time.time()
                self.waiting_for_landmark_output = True
            
            # Check shutdown before publishing
            if not self.is_shutting_down:
                try:
                    self.landmark_detector_input_tensor_pub.publish(tensor_msg)
                    self.get_logger().info(f"[{self.current_request_id}] Published landmark_detector_input_tensor at {time.time():.6f}")
                except Exception:
                    pass

        except Exception as e:
            try:
                self.get_logger().error(f"Error processing palm detector output: {e}")
            except Exception:
                pass
            with self.lock:
                self.processing = False
                self.waiting_for_palm_output = False
                self.waiting_for_landmark_output = False
                self.current_request_id = None

    def landmark_result_callback(self, msg):
        # Check shutdown flag first to avoid using destroyed resources
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
                try:
                    self.get_logger().debug("Not waiting for landmark output, skipping this callback.")
                except Exception:
                    pass
                return
            self.waiting_for_landmark_output = False

        try:
            self.get_logger().debug("Received TensorList on landmark_detector_output_tensor")
        except Exception:
            pass

        if len(msg.tensor_list) != 3:
            try:
                self.get_logger().error(f'Expected landmark tensor_list size 3, got {len(msg.tensor_list)}')
                for i, t in enumerate(msg.tensor_list):
                    try:
                        self.get_logger().error(f" Landmark Tensor[{i}]: name={t.name}, dtype={t.data_type}, shape={t.shape}, data_len={len(t.data) if t.data else 0}")
                    except Exception:
                        self.get_logger().error(f" Landmark Tensor[{i}]: unable to read meta")
            except Exception:
                pass
            with self.lock:
                self.processing = False
                self.waiting_for_palm_output = False
                self.waiting_for_landmark_output = False
                self.current_request_id = None
            return
        try:
            # Log tensor metadata for correlation
            try:
                for i, t in enumerate(msg.tensor_list):
                    self.get_logger().debug(f"[{self.current_request_id}] landmark output Tensor[{i}]: name={t.name}, dtype={t.data_type}, shape={t.shape}, data_len={len(t.data) if t.data else 0}")
            except Exception:
                pass

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

            with self.lock:
                self.processing = False
            
            # Publish the processed image - check shutdown first
            if not self.is_shutting_down:
                try:
                    self.get_logger().info(f"[{self.current_request_id}] Publishing processed image at {time.time():.6f}")
                    self.publisher_.publish(processed_msg)
                except Exception:
                    pass

            with self.lock:
                self.current_request_id = None
        except Exception as e:
            try:
                self.get_logger().error(f"Error processing palm detector output: {e}")
            except Exception:
                pass
            with self.lock:
                self.processing = False
                self.waiting_for_palm_output = False
                self.waiting_for_landmark_output = False
                self.current_request_id = None

def main(args=None):
    rclpy.init(args=args)
    hand_detector_node = None
    executor = None
    try:
        hand_detector_node = HandLandmarkDetector()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(hand_detector_node)
        executor.spin()
    except KeyboardInterrupt:
        print("HandLandmarkDetector node stopped by user")
    except Exception as e:
        print(f"Error in HandLandmarkDetector: {e}")
    finally:
        # Set shutdown flag to prevent callbacks from executing during cleanup
        if hand_detector_node is not None:
            hand_detector_node.is_shutting_down = True

        # Shutdown executor first to stop spinning
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=2.0)
            except Exception as e:
                print(f"Executor shutdown error (non-fatal): {e}")

        # Destroy node after executor is stopped
        if hand_detector_node is not None:
            try:
                hand_detector_node.destroy_node()
            except Exception as e:
                print(f"Node destruction error (non-fatal): {e}")

        # Finally shutdown rclpy context
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as e:
                print(f"rclpy shutdown error (non-fatal): {e}")

if __name__ == '__main__':
    main()
