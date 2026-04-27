# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import ast
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .geometry import quaternion_from_matrix, quaternion_matrix

OBJLIST = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
BORDER_LIST = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass
class CameraIntrinsics:
    # orgin LineMOD intrinsics
    # cam_fx: float = 572.41140
    # cam_fy: float = 573.57043
    # cam_cx: float = 325.26110
    # cam_cy: float = 242.04899
    cam_fx: float = 461.07720947265625
    cam_fy: float = 461.29638671875
    cam_cx: float = 318.0372009277344
    cam_cy: float = 236.3270721435547


def mask_to_bbox(mask: np.ndarray):
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = y = w = h = 0
    for contour in contours:
        tx, ty, tw, th = cv2.boundingRect(contour)
        if tw * th > w * h:
            x, y, w, h = tx, ty, tw, th
    return [x, y, w, h]


def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    bbx[0] = max(bbx[0], 0)
    bbx[1] = min(bbx[1], 479)
    bbx[2] = max(bbx[2], 0)
    bbx[3] = min(bbx[3], 639)
    rmin, rmax, cmin, cmax = bbx
    r_b = rmax - rmin
    c_b = cmax - cmin
    for idx in range(len(BORDER_LIST) - 1):
        if BORDER_LIST[idx] < r_b < BORDER_LIST[idx + 1]:
            r_b = BORDER_LIST[idx + 1]
            break
    for idx in range(len(BORDER_LIST) - 1):
        if BORDER_LIST[idx] < c_b < BORDER_LIST[idx + 1]:
            c_b = BORDER_LIST[idx + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        rmax += -rmin
        rmin = 0
    if cmin < 0:
        cmax += -cmin
        cmin = 0
    if rmax > 480:
        rmin -= rmax - 480
        rmax = 480
    if cmax > 640:
        cmin -= cmax - 640
        cmax = 640
    return rmin, rmax, cmin, cmax


class OnnxDenseFusion:
    def __init__(self, pose_onnx_path: str, refine_onnx_path: str):
        self.pose_sess = ort.InferenceSession(pose_onnx_path, providers=["CPUExecutionProvider"])
        self.refine_sess = ort.InferenceSession(refine_onnx_path, providers=["CPUExecutionProvider"])


class FastSamOnnx:
    def __init__(self, fastsam_onnx_path: str, score_th: float = 0.4, mask_th: float = 0.5):
        self.model_path = fastsam_onnx_path
        self.score_th = score_th
        self.mask_th = mask_th
        self._ensure_external_data()
        self.sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

    def _ensure_external_data(self) -> None:
        model_dir = os.path.dirname(self.model_path)
        default_data_path = os.path.join(model_dir, "model.data")
        if os.path.exists(default_data_path):
            return
        base_name = os.path.splitext(os.path.basename(self.model_path))[0]
        candidate = os.path.join(model_dir, f"{base_name}.data")
        if os.path.exists(candidate):
            try:
                os.symlink(candidate, default_data_path)
            except FileExistsError:
                pass

    def infer_mask(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        h, w = rgb.shape[:2]
        rgb_640 = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        inp = np.transpose(rgb_640.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, :, :, :]
        boxes, scores, coeffs, protos = self.sess.run(None, {"image": inp})
        scores = scores[0]
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) < self.score_th:
            return None
        box = boxes[0, best_idx].astype(np.int32)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1 = max(0, min(639, x1))
        y1 = max(0, min(639, y1))
        x2 = max(0, min(639, x2))
        y2 = max(0, min(639, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        proto = protos[0]  # [32, 160, 160]
        coeff = coeffs[0, best_idx]  # [32]
        lowres = np.tensordot(coeff, proto, axes=(0, 0))
        lowres = 1.0 / (1.0 + np.exp(-lowres))
        mask_640 = cv2.resize(lowres.astype(np.float32), (640, 640), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_640 > self.mask_th).astype(np.uint8)
        box_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = 1
        mask_bin = (mask_bin * box_mask).astype(np.uint8) * 255
        if mask_bin.sum() == 0:
            return None
        return cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)


class Yolo11SegOnnx:
    def __init__(self, yolo_onnx_path: str, score_th: float = 0.25, mask_th: float = 0.5):
        self.score_th = score_th
        self.mask_th = mask_th
        self.names = COCO_NAMES
        self.alias = {"duck": "bird"}
        self.sess = ort.InferenceSession(yolo_onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        in_shape = self.sess.get_inputs()[0].shape
        self.input_h = int(in_shape[2]) if len(in_shape) >= 4 and isinstance(in_shape[2], int) else 640
        self.input_w = int(in_shape[3]) if len(in_shape) >= 4 and isinstance(in_shape[3], int) else 640
        self._load_names_from_metadata()

    def _load_names_from_metadata(self) -> None:
        try:
            meta = self.sess.get_modelmeta().custom_metadata_map
            names_raw = meta.get("names")
            if not names_raw:
                return
            names_obj = ast.literal_eval(names_raw)
            if isinstance(names_obj, dict):
                ordered = []
                for key in sorted(names_obj.keys(), key=lambda k: int(k)):
                    ordered.append(str(names_obj[key]))
                if ordered:
                    self.names = ordered
        except Exception:
            # Keep default COCO names when metadata parsing fails.
            pass

    @staticmethod
    def _letterbox(
        rgb: np.ndarray,
        new_shape: Tuple[int, int],
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
        h, w = rgb.shape[:2]
        new_h, new_w = new_shape
        r = min(float(new_w) / float(w), float(new_h) / float(h))
        resize_w = int(round(w * r))
        resize_h = int(round(h * r))
        resized = cv2.resize(rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        dw = new_w - resize_w
        dh = new_h - resize_h
        left = int(dw // 2)
        right = int(dw - left)
        top = int(dh // 2)
        bottom = int(dh - top)
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, r, (left, top, right, bottom)

    @staticmethod
    def _deletterbox_mask(mask_lb: np.ndarray, pads: Tuple[int, int, int, int], out_h: int, out_w: int) -> np.ndarray:
        left, top, right, bottom = pads
        h, w = mask_lb.shape[:2]
        y0 = max(0, top)
        y1 = max(y0, h - bottom)
        x0 = max(0, left)
        x1 = max(x0, w - right)
        unpadded = mask_lb[y0:y1, x0:x1]
        if unpadded.size == 0:
            return np.zeros((out_h, out_w), dtype=np.uint8)
        return cv2.resize(unpadded, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    def _target_class_id(self, target_label: str) -> Optional[int]:
        label = target_label.lower().strip()
        if label in self.names:
            return self.names.index(label)
        if label in self.alias and self.alias[label] in self.names:
            return self.names.index(self.alias[label])
        return None

    def infer_mask(self, rgb: np.ndarray, target_label: str = "duck") -> Optional[np.ndarray]:
        h, w = rgb.shape[:2]
        rgb_lb, _, pads = self._letterbox(rgb, (self.input_h, self.input_w))
        inp = np.transpose(rgb_lb.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, :, :, :]
        dets, protos = self.sess.run(None, {self.input_name: inp})
        dets = dets[0]  # [300, 38]
        target_id = self._target_class_id(target_label)
        if target_id is None:
            return None

        keep = []
        for row in dets:
            score = float(row[4])
            cls_id = int(row[5])
            if score < self.score_th:
                continue
            if cls_id != target_id:
                continue
            keep.append(row)
        if not keep:
            return None
        best = max(keep, key=lambda r: float(r[4]))

        x1, y1, x2, y2 = [int(v) for v in best[:4]]
        x1, y1 = max(0, min(self.input_w - 1, x1)), max(0, min(self.input_h - 1, y1))
        x2, y2 = max(0, min(self.input_w - 1, x2)), max(0, min(self.input_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        coeff = best[6:].astype(np.float32)  # [32]
        proto = protos[0]  # [32, 160, 160]
        lowres = np.tensordot(coeff, proto, axes=(0, 0))
        lowres = 1.0 / (1.0 + np.exp(-lowres))
        mask_lb = cv2.resize(lowres.astype(np.float32), (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_lb > self.mask_th).astype(np.uint8)
        box_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = 1
        mask_bin = (mask_bin * box_mask).astype(np.uint8) * 255
        if mask_bin.sum() == 0:
            return None
        mask = self._deletterbox_mask(mask_bin, pads, h, w)
        if mask.sum() == 0:
            return None
        return mask

    def infer_instances(self, rgb: np.ndarray, score_th: Optional[float] = None):
        h, w = rgb.shape[:2]
        th = self.score_th if score_th is None else score_th
        rgb_lb, _, pads = self._letterbox(rgb, (self.input_h, self.input_w))
        inp = np.transpose(rgb_lb.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, :, :, :]
        dets, protos = self.sess.run(None, {self.input_name: inp})
        dets = dets[0]
        proto = protos[0]
        instances = []
        for row in dets:
            score = float(row[4])
            if score < th:
                continue
            cls_id = int(row[5])
            x1, y1, x2, y2 = [int(v) for v in row[:4]]
            x1, y1 = max(0, min(self.input_w - 1, x1)), max(0, min(self.input_h - 1, y1))
            x2, y2 = max(0, min(self.input_w - 1, x2)), max(0, min(self.input_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            coeff = row[6:].astype(np.float32)
            lowres = np.tensordot(coeff, proto, axes=(0, 0))
            lowres = 1.0 / (1.0 + np.exp(-lowres))
            mask_lb = cv2.resize(lowres.astype(np.float32), (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_lb > self.mask_th).astype(np.uint8)
            box_mask = np.zeros_like(mask_bin, dtype=np.uint8)
            box_mask[y1:y2, x1:x2] = 1
            mask_bin = (mask_bin * box_mask).astype(np.uint8) * 255
            if mask_bin.sum() == 0:
                continue
            mask = self._deletterbox_mask(mask_bin, pads, h, w)
            if mask.sum() == 0:
                continue
            instances.append(
                {
                    "class_id": cls_id,
                    "label": self.names[cls_id] if 0 <= cls_id < len(self.names) else str(cls_id),
                    "score": score,
                    "bbox_xyxy_640": [x1, y1, x2, y2],
                    "mask": mask,
                }
            )
        return instances


def preprocess_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    obj_index: int,
    num_points: int,
    intr: CameraIntrinsics,
    input_h: int = 80,
    input_w: int = 80,
) -> Optional[Dict[str, np.ndarray]]:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    valid_mask = (mask != 0) & (depth != 0)
    if valid_mask.sum() == 0:
        return None

    bbox = mask_to_bbox((mask != 0).astype(np.uint8))
    rmin, rmax, cmin, cmax = get_bbox(bbox)
    rgb_crop = rgb[rmin:rmax, cmin:cmax, :3]
    depth_crop = depth[rmin:rmax, cmin:cmax]
    mask_crop = (mask[rmin:rmax, cmin:cmax] != 0).astype(np.uint8)

    if rgb_crop.size == 0 or depth_crop.size == 0:
        return None

    rgb_resized = cv2.resize(rgb_crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth_crop, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
    mask_resized = cv2.resize(mask_crop, (input_w, input_h), interpolation=cv2.INTER_NEAREST)

    choose = ((mask_resized != 0) & (depth_resized > 0)).flatten().nonzero()[0]
    if choose.size == 0:
        return None
    if choose.size > num_points:
        c_mask = np.zeros(choose.size, dtype=np.int32)
        c_mask[:num_points] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, num_points - choose.size), mode="wrap")

    # Map resized crop pixels back to original image coordinates for correct 3D back-projection.
    if input_h > 1:
        row_coords = np.linspace(rmin, rmax - 1, input_h, dtype=np.float32)
    else:
        row_coords = np.array([float(rmin)], dtype=np.float32)
    if input_w > 1:
        col_coords = np.linspace(cmin, cmax - 1, input_w, dtype=np.float32)
    else:
        col_coords = np.array([float(cmin)], dtype=np.float32)
    xmap = np.tile(row_coords.reshape(-1, 1), (1, input_w))
    ymap = np.tile(col_coords.reshape(1, -1), (input_h, 1))
    depth_masked = depth_resized.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
    pt2 = depth_masked
    pt0 = (ymap_masked - intr.cam_cx) * pt2 / intr.cam_fx
    pt1 = (xmap_masked - intr.cam_cy) * pt2 / intr.cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1) / 1000.0
    choose = choose.reshape(1, 1, -1).astype(np.int64)
    rgb_chw = np.transpose(rgb_resized.astype(np.float32), (2, 0, 1))
    rgb_chw = rgb_chw / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb_norm = (rgb_chw - mean) / std

    return {
        "points": cloud.astype(np.float32)[np.newaxis, :, :],
        "choose": choose,
        "img": rgb_norm[np.newaxis, :, :, :],
        "idx": np.array([obj_index], dtype=np.int64),
    }


def infer_pose_onnx(runner: OnnxDenseFusion, data: Dict[str, np.ndarray], iteration: int, num_points: int):
    pred_r, pred_t, pred_c, emb = runner.pose_sess.run(
        None,
        {
            "img": data["img"],
            "points": data["points"],
            "choose": data["choose"],
            "obj": data["idx"],
        },
    )
    pred_r_norm = np.linalg.norm(pred_r, axis=2, keepdims=True)
    pred_r_norm[pred_r_norm < 1e-8] = 1.0
    pred_r = pred_r / pred_r_norm
    pred_c = pred_c.reshape(1, num_points)
    which_max = int(np.argmax(pred_c, axis=1)[0])
    pred_t = pred_t.reshape(num_points, 1, 3)
    my_r = pred_r[0][which_max].reshape(-1)
    my_t = (data["points"].reshape(num_points, 1, 3) + pred_t)[which_max].reshape(-1)

    for _ in range(iteration):
        t_tensor = np.repeat(my_t.astype(np.float32).reshape(1, 1, 3), num_points, axis=1)
        mat = quaternion_matrix(my_r)
        r_tensor = mat[:3, :3].astype(np.float32).reshape(1, 3, 3)
        mat[0:3, 3] = my_t
        if not np.all(np.isfinite(r_tensor)) or not np.all(np.isfinite(t_tensor)):
            break
        new_points = np.matmul((data["points"] - t_tensor), r_tensor)
        pred_r_refine, pred_t_refine = runner.refine_sess.run(
            None,
            {
                "points": new_points.astype(np.float32),
                "emb": emb.astype(np.float32),
                "obj": data["idx"],
            },
        )
        pred_r_refine = pred_r_refine.reshape(1, 1, -1)
        refine_norm = np.linalg.norm(pred_r_refine, axis=2, keepdims=True)
        refine_norm[refine_norm < 1e-8] = 1.0
        pred_r_refine = pred_r_refine / refine_norm
        my_r_2 = pred_r_refine.reshape(-1)
        my_t_2 = pred_t_refine.reshape(-1)
        mat_2 = quaternion_matrix(my_r_2)
        mat_2[0:3, 3] = my_t_2
        mat_final = np.dot(mat, mat_2)
        if not np.all(np.isfinite(mat_final)):
            break
        r_final = mat_final.copy()
        r_final[0:3, 3] = 0
        my_r = quaternion_from_matrix(r_final, False)
        q_norm = np.linalg.norm(my_r)
        if not np.isfinite(q_norm) or q_norm < 1e-8:
            break
        my_r = my_r / q_norm
        my_t = np.array([mat_final[0][3], mat_final[1][3], mat_final[2][3]], dtype=np.float32)
        if not np.all(np.isfinite(my_t)):
            break
    return my_r.astype(np.float32), my_t.astype(np.float32), float(pred_c[0, which_max])
