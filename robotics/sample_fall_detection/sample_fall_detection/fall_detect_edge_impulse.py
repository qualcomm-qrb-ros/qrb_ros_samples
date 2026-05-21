# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from sensor_msgs.msg import Image

import cv2
import numpy as np
from cv_bridge import CvBridge

import sys
from rclpy.executors import ExternalShutdownException

def _inject_venv_site_packages():
    """
    If a venv is activated in the shell, VIRTUAL_ENV is set.
    This function adds <VIRTUAL_ENV>/lib/pythonX.Y/site-packages to sys.path
    so system Python can import packages installed in the venv.
    """
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        return False

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"

    candidates = [
        os.path.join(venv, "lib", pyver, "site-packages"),
        # Some distros/tools may use dist-packages naming:
        os.path.join(venv, "lib", pyver, "dist-packages"),
        os.path.join(venv, "local", "lib", pyver, "dist-packages"),
    ]

    added = False
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            added = True

    if added:
        # Can't use ROS logger here (node not created yet), so use stderr print.
        print(f"[sample_fall_detection] Injected venv site-packages from VIRTUAL_ENV={venv}", file=sys.stderr)
    return added


# pip3 install edge_impulse_linux
#from edge_impulse_linux.image import ImageImpulseRunner
try:
    from edge_impulse_linux.image import ImageImpulseRunner
except ModuleNotFoundError:
    # Try to make venv packages visible when running under ROS2 system Python
    _inject_venv_site_packages()
    from edge_impulse_linux.image import ImageImpulseRunner

class FallStateGateNode(Node):
    """
    Fall state gate node

    - Service: /fall_detection/state_control (std_srvs/SetBool)
        True  -> enable processing
        False -> disable processing

    - Subscribe: image_topic (sensor_msgs/Image, NV12)
        When enabled, cache the latest frame.

    - Timer: process loop
        - Optional Edge Impulse inference
        - Publish RViz-friendly image (bgr8/rgb8)
        - If fall detected: publish /fall_detected True and optionally save images
    """

    def __init__(self):
        super().__init__('fall_state_gate_node')

        self._enabled = False
        self._has_frame = False

        self._lock = threading.Lock()
        self._bridge = CvBridge()
        self._latest_bgr = None
        self._latest_header = None

        # ---------------- Parameters (override via launch) ----------------
        self.declare_parameter('image_topic', '/cam0_stream1')
        self.declare_parameter('state_service', '/fall_detection/state_control')
        self.declare_parameter('timer_period_sec', 0.03)

        self.declare_parameter('save_dir', '/tmp/fall_images')
        self.declare_parameter('save_on_fall_only', True)
        self.declare_parameter('save_ext', '.jpg')
        self.declare_parameter('save_once_per_frame', True)
        self.declare_parameter('fall_save_cooldown_sec', 0.0)
        self.declare_parameter('save_visualized', False)

        self.declare_parameter('republish_image', True)
        self.declare_parameter('republish_topic', '/fall_detection/image')
        self.declare_parameter('republish_encoding', 'bgr8')  # bgr8 or rgb8
        self.declare_parameter('default_frame_id', 'camera')

        self.declare_parameter('fall_pub_topic', '/fall_detected')
        self.declare_parameter('publish_false', False)

        self.declare_parameter('enable_inference', True)
        self.declare_parameter('model_path', './fall-detection-with-rb8-linux-aarch64-qnn-v1.eim')
        self.declare_parameter('ei_debug', False)

        self.declare_parameter('fall_label', 'fallen')
        self.declare_parameter('fall_threshold', 0.55)

        # ---------------- Read parameters ----------------
        self._image_topic = self.get_parameter('image_topic').value
        self._state_service = self.get_parameter('state_service').value
        self._timer_period = float(self.get_parameter('timer_period_sec').value)

        self._save_dir = self.get_parameter('save_dir').value
        self._save_on_fall_only = bool(self.get_parameter('save_on_fall_only').value)
        self._save_ext = self.get_parameter('save_ext').value
        self._save_once_per_frame = bool(self.get_parameter('save_once_per_frame').value)
        self._cooldown = float(self.get_parameter('fall_save_cooldown_sec').value)
        self._save_visualized = bool(self.get_parameter('save_visualized').value)
        os.makedirs(self._save_dir, exist_ok=True)
        self._last_saved_time = None

        self._republish_image = bool(self.get_parameter('republish_image').value)
        self._republish_topic = self.get_parameter('republish_topic').value
        self._republish_encoding = self.get_parameter('republish_encoding').value.lower()
        self._default_frame_id = self.get_parameter('default_frame_id').value

        self._fall_pub_topic = self.get_parameter('fall_pub_topic').value
        self._publish_false = bool(self.get_parameter('publish_false').value)

        self._enable_infer = bool(self.get_parameter('enable_inference').value)
        self._model_path = self.get_parameter('model_path').value
        self._ei_debug = bool(self.get_parameter('ei_debug').value)

        self._fall_label = self.get_parameter('fall_label').value
        self._fall_threshold = float(self.get_parameter('fall_threshold').value)

        # ---------------- Publishers ----------------
        self._pub_fall = self.create_publisher(Bool, self._fall_pub_topic, 10)
        self._pub_img = self.create_publisher(Image, self._republish_topic, 10)

        # ---------------- Subscriber ----------------
        self._sub_img = self.create_subscription(
            Image, self._image_topic, self.image_callback, qos_profile_sensor_data
        )

        # ---------------- Service ----------------
        self._srv = self.create_service(SetBool, self._state_service, self.state_control_cb)

        # ---------------- Timer ----------------
        self._timer = self.create_timer(self._timer_period, self.process_and_save)

        # ---------------- Edge Impulse runner ----------------
        self._runner = None
        self._model_info = None
        if self._enable_infer:
            self._init_ei_runner()

        self.get_logger().info(
            f'Node started. sub={self._image_topic}, srv={self._state_service}, '
            f'pub_fall={self._fall_pub_topic}, repub_img={self._republish_topic}({self._republish_encoding})'
        )

        self.get_logger().info(f"Python executable: {sys.executable}")
        self.get_logger().info(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', '')}")


    def state_control_cb(self, request: SetBool.Request, response: SetBool.Response):
        self._enabled = bool(request.data)
        response.success = True
        response.message = f'enabled set to {self._enabled}'
        self.get_logger().info(f'state_control={request.data} -> enabled={self._enabled}')
        return response

    def image_callback(self, msg: Image):
        if not self._enabled:
            return

        bgr = self._nv12_to_bgr(msg)
        if bgr is None:
            return

        with self._lock:
            self._latest_bgr = bgr
            self._latest_header = msg.header
            self._has_frame = True

    def _nv12_to_bgr(self, msg: Image):
        h = msg.height
        w = msg.width
        stride = msg.step

        nv12 = np.frombuffer(msg.data, dtype=np.uint8)
        expected = stride * h * 3 // 2
        if nv12.size < expected:
            self.get_logger().error(f'NV12 buffer too small: got={nv12.size}, expected>={expected}')
            return None

        nv12 = nv12[:expected].reshape((h * 3 // 2, stride))
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        bgr = bgr[:, :w, :]
        return bgr

    def _init_ei_runner(self):
        try:
            if not os.path.exists(self._model_path):
                self.get_logger().error(f'EIM model not found: {self._model_path}')
                return

            self._runner = ImageImpulseRunner(self._model_path)
            self._runner.__enter__()
            self._model_info = self._runner.init(debug=bool(self._ei_debug))

            owner = self._model_info["project"]["owner"]
            name = self._model_info["project"]["name"]
            self.get_logger().info(f'Loaded Edge Impulse model: {owner} / {name}')
        except Exception as e:
            self.get_logger().error(f'Failed to init Edge Impulse runner: {e}')
            self._runner = None
            self._model_info = None

    def _ei_infer(self, img_bgr):
        if (not self._enable_infer) or (self._runner is None):
            return False, None, None

        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            features, _ = self._runner.get_features_from_image_auto_studio_settings(img_rgb)
            res = self._runner.classify(features)

            result = res.get("result", {})
            score = None
            if "classification" in result:
                cls = result["classification"]
                score = float(cls.get(self._fall_label, 0.0))
                return (score >= self._fall_threshold), res, score

            return False, res, score
        except Exception as e:
            self.get_logger().error(f'Edge Impulse inference failed: {e}')
            return False, None, None

    def _overlay_fall_text(self, img_bgr, fall_detected: bool, score=None):
        label = "FALL" if fall_detected else "NO FALL"
        text = f"{label} {score:.2f}" if score is not None else label

        box_color = (0, 0, 255) if fall_detected else (0, 255, 0)
        text_color = (255, 255, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        margin = 10

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x1, y1 = margin, margin
        x2, y2 = x1 + tw + 2 * margin, y1 + th + 2 * margin

        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
        cv2.addWeighted(overlay, 0.35, img_bgr, 0.65, 0, img_bgr)

        org = (x1 + margin, y1 + margin + th)
        cv2.putText(img_bgr, text, org, font, font_scale, text_color, thickness, cv2.LINE_AA)
        return img_bgr

    def _publish_rviz_image(self, img_bgr, header):
        if not self._republish_image:
            return

        try:
            if self._republish_encoding == 'rgb8':
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                out_msg = self._bridge.cv2_to_imgmsg(img, encoding='rgb8')
            else:
                out_msg = self._bridge.cv2_to_imgmsg(img_bgr, encoding='bgr8')

            out_msg.header = header
            if not out_msg.header.frame_id:
                out_msg.header.frame_id = self._default_frame_id

            self._pub_img.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'Publish RViz image failed: {e}')

    def process_and_save(self):
        if not (self._enabled and self._has_frame):
            return

        with self._lock:
            if self._latest_bgr is None or self._latest_header is None:
                return
            img_bgr = self._latest_bgr.copy()
            header = self._latest_header
            if self._save_once_per_frame:
                self._has_frame = False

        fall_detected, _, score = self._ei_infer(img_bgr)

        img_vis = img_bgr.copy()
        self._overlay_fall_text(img_vis, fall_detected, score=score)
        self._publish_rviz_image(img_vis, header)

        msg = Bool()
        msg.data = bool(fall_detected)

        if fall_detected:
            self._pub_fall.publish(msg)
            self.get_logger().info('[FALL] detected')

            if self._can_save_now():
                filename = self._make_save_path()
                to_save = img_vis if self._save_visualized else img_bgr
                ok = cv2.imwrite(filename, to_save)
                if ok:
                    self._last_saved_time = self.get_clock().now()
                    self.get_logger().info(f'[FALL] saved={filename}')
                else:
                    self.get_logger().error(f'[FALL] failed to save image: {filename}')
        else:
            if self._publish_false:
                self._pub_fall.publish(msg)
            self.get_logger().info('[NO FALL]')

    def _make_save_path(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        ext = self._save_ext if self._save_ext.startswith('.') else ('.' + self._save_ext)
        return os.path.join(self._save_dir, f'fall_{ts}{ext}')

    def _can_save_now(self):
        if not self._save_on_fall_only:
            return True
        if self._cooldown <= 0.0:
            return True
        if self._last_saved_time is None:
            return True
        dt = (self.get_clock().now() - self._last_saved_time).nanoseconds / 1e9
        return dt >= self._cooldown

    def destroy_node(self):
        try:
            if self._runner is not None:
                self._runner.__exit__(None, None, None)
                self._runner = None
        except Exception as e:
            self.get_logger().warning(f'Close Edge Impulse runner exception: {e}')
        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)
    node = FallStateGateNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        # Ctrl+C: do not use node.get_logger() here in some cases,
        # because ROS context might already be shutting down.
        print("[sample_fall_detection] Ctrl+C received, shutting down...", file=sys.stderr)

    except ExternalShutdownException:
        # Raised when shutdown happens externally while spinning
        print("[sample_fall_detection] External shutdown requested, exiting...", file=sys.stderr)

    finally:
        # Ensure resources are released (Edge Impulse runner etc.)
        try:
            node.destroy_node()
        except Exception:
            pass

        # Safer than rclpy.shutdown() in cases where shutdown already happened
        try:
            rclpy.try_shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
