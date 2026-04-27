# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import glob
import os
from typing import List, Tuple
from pathlib import Path
import yaml

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

_RGB_EXTS = (".jpg", ".jpeg", ".png")
_DEPTH_EXTS = (".png", ".tiff", ".tif", ".jpg")


def _get_grasp_perception_package_anchor() -> Path:
    """Return a stable anchor path for package-relative lookup."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share_dir = Path(get_package_share_directory("grasp_perception")).resolve()
        if share_dir.exists():
            return share_dir
    except Exception:
        pass

    file_path = Path(__file__).resolve()
    for parent in [file_path.parent, *file_path.parents]:
        if parent.name == "grasp_perception" and (parent / "package.xml").is_file():
            return parent
    return file_path.parent


def _load_file_input_config_from_yaml() -> dict:
    """Load file_input_publisher defaults from config YAML."""
    config_path = _get_grasp_perception_package_anchor() / "config" / "file_input_publisher.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"File input config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    scoped = data.get("file_input_publisher", {}).get("ros__parameters", {})
    if not isinstance(scoped, dict) or not scoped:
        raise ValueError(
            f"Invalid config format in {config_path}: "
            "expected 'file_input_publisher.ros__parameters' as a non-empty mapping"
        )
    return scoped


class FileInputPublisher(Node):
    """Publish RGB + depth images to ROS topics.

    Two input modes are supported:
    - Directory mode (recommended): set ``scene_dir`` and scan frames from
      ``<scene_dir>/rgb/`` and ``<scene_dir>/depth/`` by sorted filename.
    - Single-frame mode (legacy-compatible): set both ``rgb_path`` and
      ``depth_path`` to repeatedly publish the same frame.
    """

    def __init__(self):
        super().__init__("file_input_publisher")

        config_defaults = _load_file_input_config_from_yaml()
        for param_name, default_value in config_defaults.items():
            self.declare_parameter(param_name, default_value)

        scene_dir   = self.get_parameter("scene_dir").value.strip()
        rgb_path    = self.get_parameter("rgb_path").value.strip()
        depth_path  = self.get_parameter("depth_path").value.strip()
        self.rgb_topic    = self.get_parameter("rgb_topic").value
        self.depth_topic  = self.get_parameter("depth_topic").value
        self.publish_hz   = float(self.get_parameter("publish_hz").value)
        self.loop         = bool(self.get_parameter("loop").value)

        # Build frame list.
        self.frames: List[Tuple[str, str]] = []  # [(rgb_path, depth_path), ...]

        if scene_dir:
            self._load_scene_dir(scene_dir)
        elif rgb_path and depth_path:
            self._load_single(rgb_path, depth_path)
        else:
            raise ValueError(
                "You must provide 'scene_dir', or both 'rgb_path' and 'depth_path'."
            )

        if not self.frames:
            raise RuntimeError("No valid RGB/depth frame pairs found. Check your paths.")

        self.frame_idx = 0

        self.rgb_pub   = self.create_publisher(Image, self.rgb_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self.timer = self.create_timer(max(0.01, 1.0 / self.publish_hz), self._tick)

        self.get_logger().info(
            f"FileInputPublisher: {len(self.frames)} frames, "
            f"publishing to {self.rgb_topic} / {self.depth_topic}, "
            f"{self.publish_hz:.1f} Hz, loop={self.loop}"
        )

    def _load_single(self, rgb_path: str, depth_path: str) -> None:
        """Single-frame mode: validate files and repeat this frame on each tick."""
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB file does not exist: {rgb_path}")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth file does not exist: {depth_path}")
        # Keep legacy behavior: single-frame mode always loops.
        self.loop = True
        self.frames.append((rgb_path, depth_path))
        self.get_logger().info(f"Single-frame mode: {os.path.basename(rgb_path)}")

    def _load_scene_dir(self, scene_dir: str) -> None:
        """Directory mode: match same-stem frames under rgb/ and depth/."""
        rgb_dir   = os.path.join(scene_dir, "rgb")
        depth_dir = os.path.join(scene_dir, "depth")

        if not os.path.isdir(rgb_dir):
            raise FileNotFoundError(f"RGB subdirectory does not exist: {rgb_dir}")
        if not os.path.isdir(depth_dir):
            raise FileNotFoundError(f"Depth subdirectory does not exist: {depth_dir}")

        # Collect RGB files and sort by filename.
        rgb_files: List[str] = []
        for ext in _RGB_EXTS:
            rgb_files.extend(glob.glob(os.path.join(rgb_dir, f"*{ext}")))
        rgb_files = sorted(set(rgb_files), key=os.path.basename)

        skipped = 0
        for rgb_file in rgb_files:
            stem = os.path.splitext(os.path.basename(rgb_file))[0]
            depth_file = self._find_depth(depth_dir, stem)
            if depth_file is None:
                self.get_logger().warn(
                    f"RGB {os.path.basename(rgb_file)} has no matching depth file; skipped."
                )
                skipped += 1
                continue
            self.frames.append((rgb_file, depth_file))

        self.get_logger().info(
            f"Directory mode: scene_dir={scene_dir}, "
            f"{len(self.frames)} frames loaded, {skipped} skipped."
        )

    @staticmethod
    def _find_depth(depth_dir: str, stem: str) -> str | None:
        """Find a matching depth file in ``depth_dir`` by stem."""
        for ext in _DEPTH_EXTS:
            candidate = os.path.join(depth_dir, f"{stem}{ext}")
            if os.path.exists(candidate):
                return candidate
        return None

    # Image helpers.

    @staticmethod
    def _to_image_msg(arr: np.ndarray, encoding: str, stamp, frame_id: str) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(arr.shape[0])
        msg.width  = int(arr.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = int(arr.strides[0])
        msg.data = np.ascontiguousarray(arr).tobytes()
        return msg

    @staticmethod
    def _ensure_depth_u16(depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            return depth
        if depth.dtype in (np.float32, np.float64):
            depth_mm = depth.copy()
            if float(depth_mm.max()) < 100.0:   # Likely meters, convert to mm.
                depth_mm = depth_mm * 1000.0
            return depth_mm.clip(0.0, 65535.0).astype(np.uint16)
        return depth.astype(np.uint16)

    # Timer callback.

    def _tick(self) -> None:
        if self.frame_idx >= len(self.frames):
            if self.loop:
                self.frame_idx = 0
                self.get_logger().info("All frames played. Looping back to frame 1.")
            else:
                self.get_logger().info("All frames played. Stopping publisher.")
                self.timer.cancel()
                return

        rgb_path, depth_path = self.frames[self.frame_idx]

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb is None or depth is None:
            self.get_logger().warn(
                f"Failed to load frame {self.frame_idx + 1}; skipped: {rgb_path}"
            )
            self.frame_idx += 1
            return

        depth = self._ensure_depth_u16(depth)

        if depth.shape[:2] != rgb.shape[:2]:
            self.get_logger().warn(
                f"Resolution mismatch at frame {self.frame_idx + 1}: "
                f"rgb={rgb.shape[:2]} depth={depth.shape[:2]}; skipped."
            )
            self.frame_idx += 1
            return

        stamp = self.get_clock().now().to_msg()
        self.rgb_pub.publish(
            self._to_image_msg(rgb, "bgr8", stamp, "camera_color_frame")
        )
        self.depth_pub.publish(
            self._to_image_msg(depth, "16UC1", stamp, "camera_depth_frame")
        )

        self.get_logger().info(
            f"[{self.frame_idx + 1}/{len(self.frames)}] {os.path.basename(rgb_path)}"
        )
        self.frame_idx += 1


def main():
    rclpy.init()
    node = FileInputPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
