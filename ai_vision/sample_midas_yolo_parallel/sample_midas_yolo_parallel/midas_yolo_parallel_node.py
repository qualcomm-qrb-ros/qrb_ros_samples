# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import ctypes
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from qrb_ros_tensor_list_msgs.msg import Tensor, TensorList
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image


# ---------------------------------------------------------------------------
# RpcMem pool — pre-allocated ION buffers for zero-copy DMA-BUF tensor send.
# Falls back gracefully if libcdsprpc is unavailable.
# ---------------------------------------------------------------------------

class _RpcMemLib:
    """Thin ctypes wrapper around libcdsprpc rpcmem_alloc / rpcmem_to_fd / rpcmem_free."""
    HEAP_ID_SYSTEM = 25
    FLAGS_DEFAULT  = 1

    def __init__(self):
        try:
            self._lib = ctypes.CDLL('libcdsprpc.so', use_errno=True)
            self._lib.rpcmem_alloc.restype  = ctypes.c_void_p
            self._lib.rpcmem_alloc.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_int]
            self._lib.rpcmem_to_fd.restype  = ctypes.c_int
            self._lib.rpcmem_to_fd.argtypes = [ctypes.c_void_p]
            self._lib.rpcmem_free.restype   = None
            self._lib.rpcmem_free.argtypes  = [ctypes.c_void_p]
            self.available = True
        except OSError:
            self.available = False

    def alloc(self, size: int):
        """Returns (ptr: c_void_p, fd: int) or (None, -1) on failure."""
        if not self.available:
            return None, -1
        ptr = self._lib.rpcmem_alloc(self.HEAP_ID_SYSTEM, self.FLAGS_DEFAULT, size)
        if not ptr:
            return None, -1
        fd = self._lib.rpcmem_to_fd(ptr)
        if fd < 0:
            self._lib.rpcmem_free(ptr)
            return None, -1
        return ptr, fd

    def free(self, ptr):
        if self.available and ptr:
            self._lib.rpcmem_free(ptr)


_rpcmem = _RpcMemLib()


@dataclass
class _RpcBuf:
    """One pre-allocated rpcmem buffer."""
    ptr:  int    # c_void_p value
    fd:   int
    size: int
    in_use: bool = False

    def write(self, data: np.ndarray) -> None:
        """Copy numpy array into the ION buffer via ctypes memmove."""
        raw = data.tobytes()
        ctypes.memmove(self.ptr, raw, len(raw))


class RpcMemPool:
    """
    Fixed pool of rpcmem ION buffers for a single tensor slot.
    Provides acquire() / release(fd) so frames can be in-flight concurrently.
    Falls back to returning fd=-1 when rpcmem is unavailable.
    """
    def __init__(self, size: int, slots: int):
        self._lock   = threading.Lock()
        self._size   = size
        self._bufs: List[_RpcBuf] = []
        if _rpcmem.available:
            for _ in range(slots):
                ptr, fd = _rpcmem.alloc(size)
                if ptr and fd >= 0:
                    self._bufs.append(_RpcBuf(ptr=ptr, fd=fd, size=size))

    def acquire(self) -> Optional[_RpcBuf]:
        """Return a free buffer, or None if none available (caller falls back to data copy)."""
        with self._lock:
            for b in self._bufs:
                if not b.in_use:
                    b.in_use = True
                    return b
        return None

    def release(self, fd: int) -> None:
        with self._lock:
            for b in self._bufs:
                if b.fd == fd:
                    b.in_use = False
                    return

    def __del__(self):
        for b in self._bufs:
            _rpcmem.free(ctypes.c_void_p(b.ptr))
        self._bufs.clear()


COCO_CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
)

SEG_COLORS = (
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
)


@dataclass
class Detection:
    cls: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    coeff: np.ndarray


@dataclass
class PendingFrame:
    header_sec: int
    header_nsec: int
    image_bgr: np.ndarray
    yolo_input_w: int
    yolo_input_h: int
    midas_tensors: Optional[List[Tensor]] = None
    yolo_tensors: Optional[List[Tensor]] = None
    created_at: float = 0.0
    # DMA-BUF fds in use for this frame — released after fusion
    midas_dmabuf_fd: int = -1
    yolo_dmabuf_fd: int = -1


def _to_float01(data: np.ndarray) -> np.ndarray:
    if data.dtype.kind == 'f':
        return data.astype(np.float32)
    if data.dtype == np.uint8:
        return data.astype(np.float32) / 255.0
    if data.dtype == np.uint16:
        return data.astype(np.float32) / 65535.0
    if data.dtype == np.int8:
        return np.clip((data.astype(np.float32) + 128.0) / 255.0, 0.0, 1.0)
    if data.dtype == np.int16:
        return np.clip((data.astype(np.float32) + 32768.0) / 65535.0, 0.0, 1.0)
    max_abs = float(np.max(np.abs(data))) if data.size else 1.0
    if max_abs < 1e-6:
        return data.astype(np.float32)
    return data.astype(np.float32) / max_abs


def _iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    denom = max(area_a + area_b - inter, 1e-6)
    return inter / denom


def _nms(dets: List[Detection], iou_thresh: float) -> List[Detection]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for det in dets:
        keep = True
        for k in kept:
            if det.cls == k.cls and _iou(det, k) > iou_thresh:
                keep = False
                break
        if keep:
            kept.append(det)
    return kept


class MidasYoloParallelNode(Node):
    def __init__(self):
        super().__init__('midas_yolo_parallel_node')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.pending: Dict[Tuple[int, int], PendingFrame] = {}
        self.synthetic_seq = 0

        self.declare_parameter('input_topic', '/image_raw')
        self.declare_parameter('midas_input_tensor_name', 'image')
        self.declare_parameter('yolo_input_tensor_name', 'image')
        self.declare_parameter('midas_tensor_data_type', 0)
        self.declare_parameter('yolo_tensor_data_type', 0)
        self.declare_parameter('yolo_pack_uint16_input', True)
        self.declare_parameter('midas_input_size', [256, 256])
        self.declare_parameter('yolo_input_size', [640, 640])
        self.declare_parameter('score_thresh', 0.25)
        self.declare_parameter('iou_thresh', 0.45)
        self.declare_parameter('overlay_alpha', 0.45)
        self.declare_parameter('max_pending_frames', 4)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.midas_input_name = str(self.get_parameter('midas_input_tensor_name').value)
        self.yolo_input_name = str(self.get_parameter('yolo_input_tensor_name').value)
        self.midas_tensor_data_type = int(self.get_parameter('midas_tensor_data_type').value)
        self.yolo_tensor_data_type = int(self.get_parameter('yolo_tensor_data_type').value)
        self.yolo_pack_uint16_input = bool(self.get_parameter('yolo_pack_uint16_input').value)
        self.midas_size = tuple(int(v) for v in self.get_parameter('midas_input_size').value)
        self.yolo_size = tuple(int(v) for v in self.get_parameter('yolo_input_size').value)
        self.score_thresh = float(self.get_parameter('score_thresh').value)
        self.iou_thresh = float(self.get_parameter('iou_thresh').value)
        self.overlay_alpha = float(self.get_parameter('overlay_alpha').value)
        self.max_pending_frames = int(self.get_parameter('max_pending_frames').value)

        self.image_sub = self.create_subscription(Image, self.input_topic, self.image_callback, 10)

        # Reentrant group so midas and yolo output callbacks can fire concurrently
        # under MultiThreadedExecutor when both results arrive at the same time.
        self._output_cb_group = ReentrantCallbackGroup()
        self.midas_out_sub = self.create_subscription(
            TensorList, 'midas_inference_output_tensor', self.midas_output_callback, 10,
            callback_group=self._output_cb_group,
        )
        self.yolo_out_sub = self.create_subscription(
            TensorList, 'yolo_seg_inference_output_tensor', self.yolo_output_callback, 10,
            callback_group=self._output_cb_group,
        )

        self.midas_in_pub = self.create_publisher(TensorList, 'midas_inference_input_tensor', 10)
        self.yolo_in_pub = self.create_publisher(TensorList, 'yolo_seg_inference_input_tensor', 10)
        self.overlay_pub = self.create_publisher(Image, 'midas_yolo_overlay', 10)
        self.depth_color_pub = self.create_publisher(Image, 'midas_depth_map', 10)
        self.depth_gray_pub = self.create_publisher(Image, 'midas_depth_gray', 10)

        self.last_log = time.time()
        self.processed_count = 0

        # Thread pool for fusion post-processing — keeps output callbacks non-blocking
        # so the next inference result can be received immediately.
        self._fuse_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2,
                                                                thread_name_prefix='fuse')

        # Pre-allocate rpcmem ION buffer pools for zero-copy DMA-BUF tensor send.
        # Pool size = max_pending_frames so all in-flight frames can hold buffers simultaneously.
        midas_bytes = int(np.prod(self.midas_size)) * 3  # HxWx3 uint8
        yolo_h, yolo_w = self.yolo_size
        if self.yolo_pack_uint16_input:
            yolo_bytes = yolo_h * yolo_w * 3 * 2  # uint16
        elif self.yolo_tensor_data_type == 2:
            yolo_bytes = yolo_h * yolo_w * 3 * 4  # float32
        else:
            yolo_bytes = yolo_h * yolo_w * 3      # uint8
        self._midas_pool = RpcMemPool(midas_bytes, self.max_pending_frames)
        self._yolo_pool  = RpcMemPool(yolo_bytes,  self.max_pending_frames)
        if _rpcmem.available and self._midas_pool._bufs and self._yolo_pool._bufs:
            self.get_logger().info(
                f'DMA-BUF pools ready: midas={len(self._midas_pool._bufs)}x{midas_bytes}B '
                f'yolo={len(self._yolo_pool._bufs)}x{yolo_bytes}B'
            )
        else:
            self.get_logger().warn(
                'rpcmem unavailable or pool alloc failed — falling back to data-copy tensors'
            )

        self.get_logger().info(
            f'midas_yolo_parallel_node started (yolo_pack_uint16_input={self.yolo_pack_uint16_input})'
        )

    def _frame_key(self, sec: int, nsec: int) -> Tuple[int, int]:
        if sec == 0 and nsec == 0:
            self.synthetic_seq += 1
            return (-1, self.synthetic_seq)
        return (sec, nsec)

    def _extract_header_key(self, msg) -> Tuple[int, int]:
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            sec = int(msg.header.stamp.sec)
            nsec = int(msg.header.stamp.nanosec)
            return self._frame_key(sec, nsec)
        self.synthetic_seq += 1
        return (-1, self.synthetic_seq)

    def _decode_image(self, msg: Image) -> Optional[np.ndarray]:
        if msg.encoding == 'nv12':
            nv12 = np.frombuffer(msg.data, dtype=np.uint8)
            yuv = nv12.reshape((msg.height * 3 // 2, msg.width))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        if msg.encoding == 'bgr8':
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if msg.encoding == 'rgb8':
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.get_logger().error(f'Unsupported encoding: {msg.encoding}')
        return None

    def _make_tensor(self, name: str, data: np.ndarray, data_type: int) -> TensorList:
        """Build a TensorList using data-copy (fallback path)."""
        msg = TensorList()
        t = Tensor()
        t.name = name
        t.data_type = data_type
        t.shape = [int(v) for v in data.shape]
        t.data = data.tobytes()
        msg.tensor_list.append(t)
        return msg

    def _make_dmabuf_tensor(
        self,
        name: str,
        data: np.ndarray,
        data_type: int,
        pool: RpcMemPool,
    ) -> Tuple[TensorList, int]:
        """
        Build a TensorList backed by a DMA-BUF ION buffer from pool.
        Returns (msg, dmabuf_fd). fd=-1 means data-copy fallback was used.
        Caller must call pool.release(fd) after the inference result arrives.
        """
        buf = pool.acquire()
        if buf is not None:
            buf.write(data)
            msg = TensorList()
            t = Tensor()
            t.name = name
            t.data_type = data_type
            t.shape = [int(v) for v in data.shape]
            t.dmabuf_fd   = buf.fd
            t.dmabuf_size = buf.size
            t.dmabuf_offset = 0
            msg.tensor_list.append(t)
            return msg, buf.fd
        # Fallback: no free buffer — use data copy
        return self._make_tensor(name, data, data_type), -1

    def _prep_midas(self, bgr: np.ndarray) -> np.ndarray:
        h, w = self.midas_size
        resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if self.midas_tensor_data_type == 2:
            rgb_f = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            rgb_f = (rgb_f - mean) / std
            return np.expand_dims(rgb_f, axis=0)
        return np.expand_dims(rgb.astype(np.uint8), axis=0)

    def _prep_yolo(self, bgr: np.ndarray) -> np.ndarray:
        h, w = self.yolo_size
        resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if self.yolo_pack_uint16_input:
            rgb_f = rgb.astype(np.float32) / 255.0
            q_scale = 0.000015259021893143654
            q = np.clip(np.rint(rgb_f / q_scale), 0.0, 65535.0).astype(np.uint16)
            return np.expand_dims(q, axis=0)
        if self.yolo_tensor_data_type == 2:
            return np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
        return np.expand_dims(rgb.astype(np.uint8), axis=0)

    def image_callback(self, msg: Image):
        bgr = self._decode_image(msg)
        if bgr is None:
            return

        key = self._extract_header_key(msg)
        sec, nsec = key

        midas_input = self._prep_midas(bgr)
        yolo_input = self._prep_yolo(bgr)

        midas_msg = self._make_tensor(self.midas_input_name, midas_input, self.midas_tensor_data_type)
        yolo_msg = self._make_tensor(self.yolo_input_name, yolo_input, self.yolo_tensor_data_type)
        if hasattr(msg, 'header'):
            midas_msg.header = msg.header
            yolo_msg.header = msg.header

        with self.lock:
            if len(self.pending) >= self.max_pending_frames:
                oldest = min(self.pending.items(), key=lambda kv: kv[1].created_at)[0]
                self.pending.pop(oldest, None)
            self.pending[key] = PendingFrame(
                header_sec=sec,
                header_nsec=nsec,
                image_bgr=bgr,
                yolo_input_w=int(self.yolo_size[1]),
                yolo_input_h=int(self.yolo_size[0]),
                created_at=time.time(),
            )

        self.midas_in_pub.publish(midas_msg)
        self.yolo_in_pub.publish(yolo_msg)

    def _tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        shape = [int(v) for v in tensor.shape]
        if not shape:
            return np.array([], dtype=np.float32)
        elem_count = int(np.prod(shape))

        # DMA-BUF output path: read directly from the ION buffer via mmap.
        if hasattr(tensor, 'dmabuf_fd') and tensor.dmabuf_fd >= 0 and tensor.dmabuf_size > 0:
            import mmap
            byte_count = tensor.dmabuf_size
            dtype = np.float32
            if byte_count == elem_count:
                dtype = np.uint8
            elif byte_count == elem_count * 2:
                dtype = np.uint16
            elif byte_count == elem_count * 4:
                dtype = np.float32
            elif byte_count == elem_count * 8:
                dtype = np.float64
            try:
                with mmap.mmap(tensor.dmabuf_fd, byte_count,
                               access=mmap.ACCESS_READ,
                               offset=int(tensor.dmabuf_offset)) as mm:
                    arr = np.frombuffer(mm, dtype=dtype, count=elem_count).copy()
                return arr.reshape(shape)
            except Exception as e:
                self.get_logger().warn(f'dmabuf mmap failed ({e}), falling back to data copy')

        # Data-copy path (fallback or non-dmabuf output).
        data = bytes(tensor.data)
        byte_count = len(data)

        dtype = np.float32
        if byte_count == elem_count:
            dtype = np.uint8
        elif byte_count == elem_count * 2:
            dtype = np.uint16
        elif byte_count == elem_count * 4:
            dtype = np.float32
        elif byte_count == elem_count * 8:
            dtype = np.float64

        arr = np.frombuffer(data, dtype=dtype, count=elem_count)
        return arr.reshape(shape)

    def _match_pending_key(self, msg: TensorList) -> Optional[Tuple[int, int]]:
        key = self._extract_header_key(msg)
        with self.lock:
            if key in self.pending:
                return key
            if len(self.pending) == 1:
                return next(iter(self.pending.keys()))
            if len(self.pending) > 1:
                return min(self.pending.items(), key=lambda kv: kv[1].created_at)[0]
        return None

    def midas_output_callback(self, msg: TensorList):
        key = self._match_pending_key(msg)
        if key is None:
            return
        frame_ready = False
        with self.lock:
            frame = self.pending.get(key)
            if frame is None:
                return
            frame.midas_tensors = list(msg.tensor_list)
            frame_ready = frame.yolo_tensors is not None
        if frame_ready:
            self._fuse_pool.submit(self._fuse_and_publish, key)

    def yolo_output_callback(self, msg: TensorList):
        key = self._match_pending_key(msg)
        if key is None:
            return
        frame_ready = False
        with self.lock:
            frame = self.pending.get(key)
            if frame is None:
                return
            frame.yolo_tensors = list(msg.tensor_list)
            frame_ready = frame.midas_tensors is not None
        if frame_ready:
            self._fuse_pool.submit(self._fuse_and_publish, key)

    def _decode_midas_depth(self, tensor: Tensor, out_w: int, out_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw = self._tensor_to_numpy(tensor)
        if raw.ndim == 4:
            raw = np.squeeze(raw, axis=0)
        if raw.ndim == 3:
            if raw.shape[0] == 1:
                raw = raw[0]
            elif raw.shape[-1] == 1:
                raw = raw[:, :, 0]
            else:
                raw = raw[:, :, 0]
        depth = raw.astype(np.float32)
        depth_resized = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        depth_gray = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_INFERNO)
        return depth_resized, depth_gray, depth_color

    def _proto_to_hwc(self, proto: np.ndarray) -> np.ndarray:
        p = proto
        if p.ndim == 4:
            p = p[0]
        if p.ndim != 3:
            raise ValueError('proto tensor rank must be 3 after squeeze')
        if p.shape[-1] == 32:
            return p.astype(np.float32)
        if p.shape[0] == 32:
            return np.transpose(p, (1, 2, 0)).astype(np.float32)
        raise ValueError('proto tensor does not have channel dimension 32')

    def _decode_yolo_legacy(self, det: np.ndarray, proto: np.ndarray, input_w: int, input_h: int) -> List[Detection]:
        if det.ndim == 4:
            det = np.squeeze(det, axis=0)
        if det.ndim == 3 and det.shape[0] == 1:
            det = det[0]
        if det.ndim != 2:
            return []

        if det.shape[0] == 116:
            det_chw = det
        elif det.shape[1] == 116:
            det_chw = det.transpose(1, 0)
        else:
            return []

        det_chw = _to_float01(det_chw) if det_chw.dtype.kind != 'f' else det_chw.astype(np.float32)
        num_anchors = det_chw.shape[1]
        dets: List[Detection] = []
        for i in range(num_anchors):
            cls_scores = det_chw[4:84, i]
            cls = int(np.argmax(cls_scores))
            score = float(cls_scores[cls])
            if score < self.score_thresh:
                continue

            cx, cy, bw, bh = [float(det_chw[k, i]) for k in range(4)]
            if bw <= 0.0 or bh <= 0.0:
                continue
            x1 = max(0.0, (cx - bw * 0.5) * input_w)
            y1 = max(0.0, (cy - bh * 0.5) * input_h)
            x2 = min(float(input_w), (cx + bw * 0.5) * input_w)
            y2 = min(float(input_h), (cy + bh * 0.5) * input_h)
            if x2 <= x1 or y2 <= y1:
                continue

            coeff = det_chw[84:116, i].astype(np.float32)
            dets.append(Detection(cls=cls, score=score, x1=x1, y1=y1, x2=x2, y2=y2, coeff=coeff))
        return _nms(dets, self.iou_thresh)

    def _decode_yolo_split(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_idx: np.ndarray,
        coeffs: np.ndarray,
        input_w: int,
        input_h: int,
    ) -> List[Detection]:
        b = np.squeeze(boxes, axis=0) if boxes.ndim == 3 and boxes.shape[0] == 1 else boxes
        s = np.squeeze(scores, axis=0) if scores.ndim == 2 and scores.shape[0] == 1 else scores
        c = np.squeeze(class_idx, axis=0) if class_idx.ndim == 2 and class_idx.shape[0] == 1 else class_idx
        m = np.squeeze(coeffs, axis=0) if coeffs.ndim == 3 and coeffs.shape[0] == 1 else coeffs

        if b.ndim != 2 or s.ndim != 1 or c.ndim != 1 or m.ndim != 2:
            return []

        if b.shape[0] == 4 and b.shape[1] != 4:
            b = b.transpose(1, 0)
        if m.shape[0] == 32 and m.shape[1] != 32:
            m = m.transpose(1, 0)

        if b.shape[1] != 4 or m.shape[1] != 32:
            return []

        n = min(b.shape[0], s.shape[0], c.shape[0], m.shape[0])
        score_vals = _to_float01(s) if s.dtype.kind != 'f' else s.astype(np.float32)

        dets: List[Detection] = []
        for i in range(n):
            score = float(score_vals[i])
            if score < self.score_thresh:
                continue

            cls = int(round(float(c[i])))
            x1, y1, x2, y2 = [float(v) for v in b[i]]
            max_box = max(abs(x1), abs(y1), abs(x2), abs(y2))
            if max_box <= 2.0:
                x1 *= input_w
                x2 *= input_w
                y1 *= input_h
                y2 *= input_h
            x1 = max(0.0, min(float(input_w), x1))
            x2 = max(0.0, min(float(input_w), x2))
            y1 = max(0.0, min(float(input_h), y1))
            y2 = max(0.0, min(float(input_h), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            coeff = m[i].astype(np.float32)
            dets.append(Detection(cls=cls, score=score, x1=x1, y1=y1, x2=x2, y2=y2, coeff=coeff))
        return _nms(dets, self.iou_thresh)

    def _parse_yolo_outputs(self, tensors: List[Tensor], input_w: int, input_h: int) -> Tuple[List[Detection], Optional[np.ndarray]]:
        arrays = []
        for t in tensors:
            arr = self._tensor_to_numpy(t)
            arrays.append((t.name, arr))

        proto = None
        det = None
        scores = None
        class_idx = None
        boxes = None
        coeffs = None

        for _, arr in arrays:
            if arr.ndim == 4:
                shp = arr.shape
                if shp[-1] == 32 or (len(shp) > 1 and shp[1] == 32):
                    proto = arr
            if arr.ndim == 3:
                shp = arr.shape
                if shp[1] == 116 or shp[2] == 116:
                    det = arr
                elif (shp[1] == 4 or shp[2] == 4) and boxes is None:
                    boxes = arr
                elif (shp[1] == 32 or shp[2] == 32) and coeffs is None:
                    coeffs = arr
            if arr.ndim == 2:
                if arr.shape[1] >= 1000 and scores is None:
                    scores = arr

        if scores is not None:
            n = int(np.squeeze(scores, axis=0).shape[0]) if scores.shape[0] == 1 else int(scores.shape[0])
            for _, arr in arrays:
                if arr.ndim != 2:
                    continue
                if arr is scores:
                    continue
                a = np.squeeze(arr, axis=0) if arr.shape[0] == 1 else arr
                if a.ndim == 1 and a.shape[0] == n and class_idx is None:
                    class_idx = arr

        if proto is None:
            return [], None
        proto_hwc = self._proto_to_hwc(proto)

        if det is not None:
            dets = self._decode_yolo_legacy(det, proto_hwc, input_w, input_h)
            return dets, proto_hwc

        if boxes is not None and scores is not None and class_idx is not None and coeffs is not None:
            dets = self._decode_yolo_split(boxes, scores, class_idx, coeffs, input_w, input_h)
            return dets, proto_hwc

        return [], proto_hwc

    def _depth_p85(self, depth: np.ndarray, mask: np.ndarray) -> float:
        vals = depth[mask > 0]
        if vals.size == 0:
            return 0.0
        return float(np.percentile(vals, 85.0))

    def _draw_overlay(
        self,
        depth_map: np.ndarray,
        depth_color: np.ndarray,
        dets: List[Detection],
        proto_hwc: Optional[np.ndarray],
        input_w: int,
        input_h: int,
    ) -> np.ndarray:
        out = depth_color.copy()
        if proto_hwc is None:
            return out

        proto_h, proto_w, _ = proto_hwc.shape
        sx = float(out.shape[1]) / float(input_w)
        sy = float(out.shape[0]) / float(input_h)
        pscale_x = float(proto_w) / float(input_w)
        pscale_y = float(proto_h) / float(input_h)

        dmin = float(np.min(depth_map))
        dmax = float(np.max(depth_map))
        dr = max(dmax - dmin, 1e-6)

        for det in dets:
            mask_logits = np.tensordot(proto_hwc, det.coeff, axes=([2], [0]))
            mask = (mask_logits >= 0.0).astype(np.uint8) * 255

            mx1 = max(0, int(det.x1 * pscale_x))
            my1 = max(0, int(det.y1 * pscale_y))
            mx2 = min(proto_w, int(det.x2 * pscale_x + 0.5))
            my2 = min(proto_h, int(det.y2 * pscale_y + 0.5))
            if mx2 <= mx1 or my2 <= my1:
                continue

            bx1 = max(0, int(det.x1 * sx))
            by1 = max(0, int(det.y1 * sy))
            bx2 = min(out.shape[1], int(det.x2 * sx + 0.5))
            by2 = min(out.shape[0], int(det.y2 * sy + 0.5))
            if bx2 <= bx1 or by2 <= by1:
                continue

            crop = mask[my1:my2, mx1:mx2]
            if crop.size == 0:
                continue
            mask_roi = cv2.resize(crop, (bx2 - bx1, by2 - by1), interpolation=cv2.INTER_NEAREST)

            color = SEG_COLORS[det.cls % len(SEG_COLORS)]
            out_roi = out[by1:by2, bx1:bx2]
            color_img = np.zeros_like(out_roi)
            color_img[:, :] = color
            blended = cv2.addWeighted(out_roi, 1.0 - self.overlay_alpha, color_img, self.overlay_alpha, 0.0)
            out_roi[mask_roi > 0] = blended[mask_roi > 0]

            class_name = COCO_CLASSES[det.cls] if 0 <= det.cls < len(COCO_CLASSES) else 'unknown'
            depth_roi = depth_map[by1:by2, bx1:bx2]
            depth_val = self._depth_p85(depth_roi, mask_roi)
            depth_norm = (depth_val - dmin) / dr
            label = f'{class_name} {det.score * 100:.0f}% d={depth_norm:.2f}'

            cv2.rectangle(out, (bx1, by1), (bx2, by2), color, 1)
            txt_sz, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            txt_x = bx1
            txt_y = min(out.shape[0] - 1, by1 + txt_sz[1] + 2)
            cv2.rectangle(out, (txt_x, by1), (min(out.shape[1] - 1, txt_x + txt_sz[0] + 2), txt_y + baseline), color, -1)
            cv2.putText(out, label, (txt_x + 1, txt_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return out

    def _fuse_and_publish(self, key: Tuple[int, int]):
        with self.lock:
            frame = self.pending.pop(key, None)
        if frame is None or not frame.midas_tensors or not frame.yolo_tensors:
            return

        try:
            midas_depth, depth_gray, depth_color = self._decode_midas_depth(
                frame.midas_tensors[0], frame.image_bgr.shape[1], frame.image_bgr.shape[0]
            )
            dets, proto = self._parse_yolo_outputs(
                frame.yolo_tensors, frame.yolo_input_w, frame.yolo_input_h
            )
            overlay = self._draw_overlay(
                midas_depth,
                depth_color,
                dets,
                proto,
                frame.yolo_input_w,
                frame.yolo_input_h,
            )

            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            depth_color_msg = self.bridge.cv2_to_imgmsg(depth_color, encoding='bgr8')
            depth_gray_msg = self.bridge.cv2_to_imgmsg(depth_gray, encoding='mono8')

            if frame.header_sec >= 0:
                overlay_msg.header.stamp.sec = frame.header_sec
                overlay_msg.header.stamp.nanosec = frame.header_nsec
                depth_color_msg.header.stamp.sec = frame.header_sec
                depth_color_msg.header.stamp.nanosec = frame.header_nsec
                depth_gray_msg.header.stamp.sec = frame.header_sec
                depth_gray_msg.header.stamp.nanosec = frame.header_nsec

            self.overlay_pub.publish(overlay_msg)
            self.depth_color_pub.publish(depth_color_msg)
            self.depth_gray_pub.publish(depth_gray_msg)

            self.processed_count += 1
            now = time.time()
            if now - self.last_log > 2.0:
                self.get_logger().info(
                    f'processed={self.processed_count} pending={len(self.pending)} detections={len(dets)}'
                )
                self.last_log = now
        except Exception as exc:
            self.get_logger().error(f'fusion failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = MidasYoloParallelNode()
    # 3 threads: one for image_callback, one for midas_output_callback,
    # one for yolo_output_callback — all can be in-flight simultaneously.
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node._fuse_pool.shutdown(wait=False)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
