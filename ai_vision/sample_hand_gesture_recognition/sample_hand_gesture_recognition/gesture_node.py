# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# qrb_ros_gesture_recognition main node.
#
# MediaPipe Hand 3-stage pipeline:
#   image -> [palm_detector.bin]          -> palm boxes          (NPU / qrb_ros_nn_inference)
#         -> [hand_landmark_detector.bin] -> 21 keypoints + L/R  (NPU / qrb_ros_nn_inference)
#         -> gesture classifier           -> 8 gesture classes   (in-node FP32 numpy)
#
# The first two stages are large conv models handed to qrb_ros_nn_inference on
# the NPU; the third stage is a tiny MLP computed locally in the node with FP32
# numpy -- the w8a8 quantization of canned_gesture_classifier loses too much
# accuracy (almost everything except Thumb_Up collapses to None), whereas FP32
# numpy is fully accurate at negligible cost. This matches the architecture of
# qrb_ros_samples/ai_vision/sample_hand_detection -- "large models on the NPU,
# post-processing math in numpy inside the node" (the palm/landmark blaze
# decode / NMS / sigmoid are likewise done in numpy inside the node).
#
# The structure follows qrb_ros_samples/ai_vision/sample_hand_detection (the
# palm + landmark stages); the stage-3 preprocessing / classification follows
# the official quic/ai-hub-models mediapipe_hand_gesture implementation.

import os
import json
import time
import threading

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSHistoryPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList

from mediapipe_hand_base import (
    nv12_to_bgr,
    resize_pad,
    denormalize_detections,
    BlazeDetector,
    BlazeLandmark,
)
from gesture_classifier import preprocess_hand_x64, GESTURE_LABELS
from gesture_fp32 import GestureClassifier
from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS

DEFAULT_TIMEOUT = 3.0
TIMEOUT_PAUSE_SECONDS = 6.0

# Threshold on the landmark model's hand-presence score: below it the ROI crop
# is not a hand (the palm detector can produce high-score boxes on faces and
# other regions), so we retry with the next candidate box.
PRESENCE_THRESH = 0.5

# Per-stage input dtype: TensorList.data_type=2 means float32 (matches
# sample_hand_detection).
DTYPE_FLOAT32 = 2


class GestureRecognitionNode(Node):
    def __init__(self):
        super().__init__('qrb_ros_gesture_recognition')

        # Publisher uses RELIABLE (compatible with both RELIABLE/BEST_EFFORT subs).
        qos_tensor = QoSProfile(
            depth=20,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        # Subscriber uses BEST_EFFORT: matches both RELIABLE and BEST_EFFORT
        # publishers. The QoS of nn_inference outputs and the camera is unknown,
        # so the subscriber side takes the most compatible setting to avoid
        # silently receiving no data.
        qos_tensor_sub = QoSProfile(
            depth=20,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        qos_image_pub = QoSProfile(depth=20)
        qos_image_sub = QoSProfile(
            depth=20,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # --- subscribe to the raw image ---
        self.image_sub = self.create_subscription(
            Image, 'image_raw', self.image_callback, qos_profile=qos_image_sub)

        # --- stage 1: palm ---
        self.palm_in_pub = self.create_publisher(
            TensorList, 'palm_detector_input_tensor', qos_profile=qos_tensor)
        self.palm_out_sub = self.create_subscription(
            TensorList, 'palm_detector_output_tensor', self.palm_result_callback, qos_profile=qos_tensor_sub)

        # --- stage 2: landmark ---
        self.landmark_in_pub = self.create_publisher(
            TensorList, 'landmark_detector_input_tensor', qos_profile=qos_tensor)
        self.landmark_out_sub = self.create_subscription(
            TensorList, 'landmark_detector_output_tensor', self.landmark_result_callback, qos_profile=qos_tensor_sub)

        # --- stage 3: gesture, in-node FP32 numpy, no NPU, no tensor round-trip ---

        # --- result publishers ---
        self.result_image_pub = self.create_publisher(
            Image, 'gesture_result_image', qos_profile=qos_image_pub)
        self.gesture_class_pub = self.create_publisher(
            String, 'gesture_class', qos_profile=qos_tensor)

        self.bridge = CvBridge()

        # state
        self.latest_image = None
        self.affine2 = None
        self.box2 = None
        self.palm_detections = None
        self.lock = threading.Lock()
        self.stage = 'idle'          # idle / palm / landmark / gesture
        self.last_request_time = 0.0
        self.paused_until = 0.0
        self.is_shutting_down = False

        # model path parameter
        self.declare_parameter('model_path', '/opt/model/')
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'model_path does not exist: {self.model_path}')

        # Quantization params: the scale/offset of each model's input/output
        # tensors. The AI Hub w8a8 model QNN context binaries have uint8
        # quantized I/O, so we (de)quantize float32 <-> uint8 here (QNN
        # convention: real=(q+offset)*scale).
        qp_path = os.path.join(self.model_path, 'quant_params.json')
        try:
            with open(qp_path) as f:
                self.qp = json.load(f)
            self.get_logger().info(f'loaded quant params from {qp_path}')
        except Exception as e:
            self.qp = {}
            self.get_logger().error(
                f'cannot load quant_params.json ({e}); (de)quantization disabled!')

        # palm detector + landmark regressor (reuse the sample's implementation)
        self.palm_detector = BlazeDetector()
        anchor_path = os.path.join(self.model_path, 'anchors_palm.npy')
        self.palm_detector.load_anchors(anchor_path)
        self.hand_regressor = BlazeLandmark()

        # stage-3 FP32 gesture classifier (in-node numpy inference)
        self.gesture = GestureClassifier(self.model_path)

        self.create_timer(1.0, self._timeout_check)
        self.get_logger().info('qrb_ros_gesture_recognition node initialized.')

    # ------------------------------------------------------------------
    def _reset(self):
        self.stage = 'idle'
        self.last_request_time = 0.0

    def _timeout_check(self):
        now = time.time()
        with self.lock:
            if self.paused_until and now >= self.paused_until:
                self.paused_until = 0.0
                self._reset()
            if self.stage != 'idle' and (now - self.last_request_time) > DEFAULT_TIMEOUT * 2:
                self.get_logger().warn(
                    f'Timeout in stage={self.stage}, resetting and pausing.')
                self.paused_until = now + TIMEOUT_PAUSE_SECONDS
                self._reset()

    # ---- quantize / dequantize (AI Hub w8a8 models have uint8 I/O) ----
    @staticmethod
    def _quantize(real, scale, offset):
        """float32 -> uint8. QNN convention real=(q+offset)*scale => q=real/scale-offset."""
        q = np.round(np.asarray(real, dtype=np.float32).reshape(-1) / scale - offset)
        return np.clip(q, 0, 255).astype(np.uint8)

    def _publish_quant_input(self, pub, model_key, tensor_name, shape, real_f32):
        """Quantize float32 to uint8 per this input tensor's quant params, then publish."""
        p = self.qp.get(model_key, {}).get('inputs', {}).get(tensor_name)
        t = Tensor()
        t.name = tensor_name
        t.shape = list(shape)
        if p is not None:
            t.data_type = 0  # uint8
            t.data = self._quantize(real_f32, float(p['scale']), float(p['offset'])).tobytes()
        else:  # no quant params: fall back to float32 (should not normally happen)
            t.data_type = DTYPE_FLOAT32
            t.data = np.asarray(real_f32, dtype=np.float32).tobytes()
        msg = TensorList()
        msg.tensor_list.append(t)
        pub.publish(msg)

    @staticmethod
    def _match_param_by_size(params, n_elems):
        hits = [p for p in params.values() if int(np.prod(p['dims'])) == n_elems]
        return hits[0] if len(hits) == 1 else None

    def _dequantize_outputs(self, msg, model_key):
        """Return a new float32 TensorList: dequantize uint8 outputs by scale/offset.

        Match quant params by tensor.name first; if QNN did not carry the
        semantic names through, fall back to a unique match by element count.
        """
        outs = self.qp.get(model_key, {}).get('outputs', {})
        new = TensorList()
        for t in msg.tensor_list:
            raw = bytes(t.data)
            if t.data_type == 0:  # uint8 -> dequantize
                p = outs.get(t.name) or self._match_param_by_size(outs, len(raw))
                u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                real = (u8 + float(p['offset'])) * float(p['scale']) if p is not None else u8
            else:  # already float32
                real = np.frombuffer(raw, dtype=np.float32)
            nt = Tensor()
            nt.name = t.name
            nt.data_type = DTYPE_FLOAT32
            nt.shape = list(t.shape)
            nt.data = real.astype(np.float32).tobytes()
            new.tensor_list.append(nt)
        return new

    # ---------------------- step 0: image received ----------------------------
    def image_callback(self, msg):
        if self.is_shutting_down:
            return
        now = time.time()
        with self.lock:
            if self.paused_until and now < self.paused_until:
                return
            if self.stage != 'idle':
                return  # previous frame still in flight, drop the current one
            self.stage = 'palm'
            self.last_request_time = now

        try:
            if msg.encoding == 'nv12':
                nv12 = np.frombuffer(msg.data, dtype=np.uint8)
                self.latest_image = nv12_to_bgr(nv12, msg.width, msg.height)
            elif msg.encoding in ('bgr8', 'rgb8'):
                self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                self.get_logger().error(f'Unsupported encoding: {msg.encoding}')
                with self.lock:
                    self._reset()
                return

            # MediaPipe models are trained on RGB; latest_image stays BGR for
            # visualization and is converted to RGB only before feeding the model.
            rgb = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
            img1, _img2, _scale, _pad = resize_pad(rgb)
            img1 = self.palm_detector.palm_detector_qnn_preprocess(img1)

            with self.lock:
                self.last_request_time = time.time()
            self._publish_quant_input(
                self.palm_in_pub, 'palm_detector', 'image',
                [1, 256, 256, 3], img1)
        except Exception as e:
            self.get_logger().error(f'image_callback error: {e}')
            with self.lock:
                self._reset()

    # ---------------------- stage 1 callback: palm result ----------------------
    def palm_result_callback(self, msg):
        if self.is_shutting_down:
            return
        with self.lock:
            if self.stage != 'palm':
                return
            self.stage = 'landmark'

        try:
            msg = self._dequantize_outputs(msg, 'palm_detector')
            img1, _img2, scale, pad = resize_pad(self.latest_image)
            normalized = self.palm_detector.palm_detector_qnn_postprocess(msg)

            dets = normalized[0] if (isinstance(normalized, list) and len(normalized) >= 1
                                     and hasattr(normalized[0], 'shape')) else None
            valid = (dets is not None and dets.ndim == 2
                     and dets.shape[0] >= 1 and dets.shape[1] == 19)
            if not valid:
                self.get_logger().info('No palm detected, publishing original image.')
                self._publish_result_image(self.latest_image, 'None')
                with self.lock:
                    self._reset()
                return

            # Collect all candidates, sorted by palm confidence (the sigmoid
            # score in column 18) in descending order. Note that
            # denormalize_detections mutates the even columns (including column
            # 18), so the ordering must be captured before denormalize.
            order = np.argsort(dets[:, 18])[::-1]
            all_dets = denormalize_detections([dets.copy()], scale, pad)
            xc, yc, roi_scale, theta = self.palm_detector.detection2roi(all_dets)
            self.cand_dets = all_dets
            self.cand_xc, self.cand_yc = np.ravel(xc), np.ravel(yc)
            self.cand_scale, self.cand_theta = np.ravel(roi_scale), np.ravel(theta)
            self.cand_order = order
            self.cand_pos = 0
            self.n_cands = int(all_dets.shape[0])

            # Feed landmark starting from the highest-score candidate; the
            # landmark presence score decides whether to move to the next
            # candidate -- this filters out palm-detector high-score false
            # positives on faces and the like (see the victory scenario).
            self._run_landmark_candidate(self.cand_pos)
        except Exception as e:
            self.get_logger().error(f'palm_result_callback error: {e}')
            with self.lock:
                self._reset()

    def _run_landmark_candidate(self, pos):
        """Crop the ROI of the pos-th sorted candidate box and publish it to the landmark model."""
        idx = int(self.cand_order[pos])
        xc = self.cand_xc[idx:idx + 1]
        yc = self.cand_yc[idx:idx + 1]
        theta = self.cand_theta[idx:idx + 1]
        roi_scale = self.cand_scale[idx:idx + 1]
        self.palm_detections = self.cand_dets[idx].reshape(1, 19)
        # MediaPipe models are trained on RGB; convert the ROI source to RGB
        # before feeding the model.
        rgb = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
        img, self.affine2, self.box2 = self.hand_regressor.extract_roi(
            rgb, xc, yc, theta, roi_scale)
        img = img.astype(np.float32)

        with self.lock:
            self.last_request_time = time.time()
        self._publish_quant_input(
            self.landmark_in_pub, 'hand_landmark_detector', 'image',
            [1, 224, 224, 3], img)

    # ------ stage 2 callback: landmark result -> stage 3 FP32 classify (terminal) ------
    def landmark_result_callback(self, msg):
        if self.is_shutting_down:
            return
        with self.lock:
            if self.stage != 'landmark':
                return

        try:
            msg = self._dequantize_outputs(msg, 'hand_landmark_detector')
            scores, lr, normalized_landmarks = self.hand_regressor.landmark_tensor_to_data(msg)
            flag = float(np.asarray(scores).reshape(-1)[0])

            # presence too low -> this ROI is not a hand; if candidates remain,
            # retry with the next one (staying in the landmark stage). This
            # branch MUST NOT call _reset, otherwise the next candidate's
            # landmark result would be dropped by the stage gate.
            if flag < PRESENCE_THRESH and (self.cand_pos + 1) < self.n_cands:
                self.cand_pos += 1
                self._run_landmark_candidate(self.cand_pos)
                return

            # none of the candidates look like a hand: report no gesture
            if flag < PRESENCE_THRESH:
                self.get_logger().info('No hand (all palm candidates below presence threshold).')
                self._publish_gesture('None', None, 0.0)
                with self.lock:
                    self._reset()
                return

            # denormalize to original-image coordinates (used by both
            # visualization and classification)
            landmarks = self.hand_regressor.denormalize_landmarks(
                normalized_landmarks.copy(), self.affine2)

            # Stage-3 preprocessing: it must use ORIGINAL-IMAGE-space landmarks
            # (which preserve the hand's true orientation), matching the official
            # mediapipe_hand_gesture/app.py::_run_landmark_detector -- it applies
            # an inverse-affine on the crop output back to the original image
            # before feeding the classifier. preprocess_hand_x64 only normalizes
            # translation+scale, NOT rotation, so orientation is a meaningful
            # feature. Using crop space by mistake (already normalized to
            # "wrist pointing down") would turn thumbs_down into the same
            # orientation as thumbs_up -> misclassification. z keeps the model's
            # native value (the affine acts on x,y only).
            hand_pts = landmarks.reshape(21, 3)
            x64_a = preprocess_hand_x64(hand_pts, lr, mirror=False)
            x64_b = preprocess_hand_x64(hand_pts, lr, mirror=True)

            # Stage-3 classify: in-node FP32 numpy MLP (avoids the w8a8
            # quantization accuracy collapse).
            gesture_id, gesture_name, gscores = self.gesture.classify(x64_a, x64_b)
            self.get_logger().info(
                f'Gesture: {gesture_name} (id={gesture_id}) '
                f'scores={np.round(np.asarray(gscores).reshape(-1), 3).tolist()}')
            self._publish_gesture(gesture_name, landmarks, flag)
            with self.lock:
                self._reset()
        except Exception as e:
            self.get_logger().error(f'landmark_result_callback error: {e}')
            with self.lock:
                self._reset()

    # ------------------------------------------------------------------
    def _publish_gesture(self, gesture_name, landmarks, flag):
        """Publish the class text + the visualized result image (keypoints / ROI / palm box / gesture text)."""
        s = String()
        s.data = gesture_name
        self.gesture_class_pub.publish(s)

        image = self.latest_image.copy()
        try:
            if landmarks is not None and flag > PRESENCE_THRESH:
                draw_landmarks(image, landmarks[0][:, :2], HAND_CONNECTIONS, size=2)
                if self.box2 is not None:
                    draw_roi(image, self.box2)
                if self.palm_detections is not None:
                    draw_detections(image, self.palm_detections)
        except Exception as e:
            self.get_logger().warn(f'draw error (non-fatal): {e}')
        self._draw_label(image, gesture_name)
        self._publish_result_image(image, gesture_name)

    def _draw_label(self, image, text):
        cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 2, cv2.LINE_AA)

    def _publish_result_image(self, image, gesture_name):
        if self.is_shutting_down:
            return
        try:
            out = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            self.result_image_pub.publish(out)
        except Exception as e:
            self.get_logger().warn(f'publish result image error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = GestureRecognitionNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in GestureRecognitionNode: {e}')
    finally:
        if node is not None:
            node.is_shutting_down = True
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=2.0)
            except Exception:
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
