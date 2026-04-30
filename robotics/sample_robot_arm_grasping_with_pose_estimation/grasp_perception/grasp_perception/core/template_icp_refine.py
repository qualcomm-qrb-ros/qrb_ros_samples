# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
=============================================================================
Complete 6D Pose Refinement Pipeline (No CAD, No Geometric Constraint)

Pipeline:
  YOLO11 Seg Mask
       │
  RGB-D → Object Point Cloud + Feature Extraction
       │
  ┌────┴─────────────────────────┐
  │  PnP Refiner (rotation)      │  ← texture feature matching
  │  Template ICP (translation)  │  ← point cloud geometry alignment
  └────┬─────────────────────────┘
       │
  Pose Fusion (weighted fusion)
       │
  Temporal Smoothing (temporal smoothing)
       │
  Final 6D Pose (R, t)

Dependencies: numpy, scipy, opencv-python-headless
  pip install numpy scipy opencv-python-headless
=============================================================================
"""

import numpy as np
import cv2
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# 0. Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RefinementResult:
    """Single frame refinement result"""
    pose: np.ndarray                    # [4x4] final pose
    R: np.ndarray = None                # [3x3] rotation matrix
    t: np.ndarray = None                # [3] translation vector
    icp_fitness: float = 0.0
    icp_rmse: float = float('inf')
    pnp_inliers: int = 0                
    pnp_reproj_error: float = float('inf')
    pnp_success: bool = False
    chosen_rotation: str = "densefusion" 
    frame_id: int = 0                  
    status: str = "unknown"            

    def __post_init__(self):
        if self.R is None:
            self.R = self.pose[:3, :3]
        if self.t is None:
            self.t = self.pose[:3, 3]


@dataclass
class PipelineConfig:
    """Pipeline configuration parameters"""
    # 相机内参
    fx: float = 461.07720947265625
    fy: float = 461.29638671875
    cx: float = 318.0372009277344
    cy: float = 236.3270721435547

    # depth parameters
    depth_scale: float = 1000.0     # mm → m
    depth_max: float = 1.0          # maximum valid depth (m)

    # ICP parameters
    icp_voxel_size: float = 0.002   # voxel downsampling (m)
    icp_max_dist: float = 0.015     # maximum correspondence distance (m)
    icp_max_iter: int = 50
    icp_tolerance: float = 1e-6
    icp_scales: list = field(
        default_factory=lambda: [0.008, 0.004, 0.002]
    )
    icp_max_dists: list = field(
        default_factory=lambda: [0.04, 0.02, 0.01]
    )
    icp_max_iters: list = field(
        default_factory=lambda: [30, 30, 50]
    )

    # PnP 参数
    feature_type: str = "orb"       # "orb" / "sift"
    max_features: int = 500
    match_ratio: float = 0.75       # Lowe's ratio test
    ransac_reproj: float = 3.0      # RANSAC reprojection threshold (px)
    ransac_iter: int = 200
    ransac_conf: float = 0.99

    # Mask parameters
    mask_dilate_px: int = 3         # mask dilation pixels

    # Smoothing parameters
    smooth_window: int = 5
    smooth_exp_decay: float = 0.5

    # 融合参数
    pnp_max_weight: float = 0.7     # PnP rotation maximum weight
    icp_fallback_fitness: float = 0.3  # ICP fallback threshold

    # Template update
    template_update: bool = True
    template_novel_dist: float = 0.004  # new point determination distance (m)

    # Denoising
    outlier_k: int = 20
    outlier_std: float = 2.0


# ═══════════════════════════════════════════════════════════════════════════
# 1. Point Cloud Utilities (纯 NumPy)
# ═══════════════════════════════════════════════════════════════════════════

class PointCloudUtils:
    """Point cloud utilities — alternative to Open3D"""

    @staticmethod
    def rgbd_to_points(
        depth: np.ndarray, mask: np.ndarray,
        fx: float, fy: float, cx: float, cy: float,
        depth_scale: float = 1000.0, depth_max: float = 1.0,
    ) -> np.ndarray:
        """RGB-D + Mask → point cloud [N, 3] (camera coordinate system)"""
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        depth_m = depth.astype(np.float64) / depth_scale
        valid = mask & (depth_m > 0.01) & (depth_m < depth_max)

        z = depth_m[valid]
        x = (u[valid].astype(np.float64) - cx) * z / fx
        y = (v[valid].astype(np.float64) - cy) * z / fy
        return np.column_stack([x, y, z])

    @staticmethod
    def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        """voxel downsampling"""
        if len(points) == 0:
            return points
        idx = np.floor(points / voxel_size).astype(np.int64)
        vox = {}
        for i in range(len(points)):
            key = (idx[i, 0], idx[i, 1], idx[i, 2])
            if key not in vox:
                vox[key] = []
            vox[key].append(points[i])
        return np.array([np.mean(v, axis=0) for v in vox.values()])

    @staticmethod
    def remove_outliers(
        points: np.ndarray, k: int = 20, std_ratio: float = 2.0
    ) -> np.ndarray:
        """statistical filtering denoising"""
        if len(points) < k + 1:
            return points
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k + 1)
        mean_dists = dists[:, 1:].mean(axis=1)
        threshold = mean_dists.mean() + std_ratio * mean_dists.std()
        return points[mean_dists < threshold]

    @staticmethod
    def estimate_normals(points: np.ndarray, k: int = 20, ref_point: Optional[np.ndarray] = None) -> np.ndarray:
        """PCA normal estimation"""
        tree = cKDTree(points)
        _, indices = tree.query(points, k=k)
        normals = np.zeros_like(points)

        for i in range(len(points)):
            nbr = points[indices[i]]
            cov = np.cov(nbr.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]

        # Orient normals so they point toward the reference point in the same frame.
        # If ref_point is None, use the origin (0,0,0) for legacy behavior.
        if ref_point is None:
            ref_point = np.zeros((3,), dtype=np.float64)
        else:
            ref_point = np.asarray(ref_point, dtype=np.float64).reshape(3)

        # Vector from point to camera/reference: (ref_point - p)
        # Want normals aligned with that direction => dot(normal, ref_point - p) > 0
        flip = np.sum(normals * (ref_point[None, :] - points), axis=1) < 0
        normals[flip] *= -1
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.clip(norms, 1e-10, None)
        return normals

    @staticmethod
    def prepare_mask(
        yolo_mask: np.ndarray, dilate_px: int = 3
    ) -> np.ndarray:
        """YOLO bool mask → OpenCV uint8 mask (with dilation)"""
        mask_uint8 = yolo_mask.astype(np.uint8) * 255
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
        return mask_uint8


# ═══════════════════════════════════════════════════════════════════════════
# 2. Template ICP Refiner (Point-to-Plane, multi-scale)
# ═══════════════════════════════════════════════════════════════════════════

class TemplateICPRefiner:
    """
    Point-to-Plane ICP refinement based on the first frame template.
    No CAD model, no geometric constraints.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.pcu = PointCloudUtils()

        # Template state
        self.template_pts = None
        self.template_normals = None
        self.initialized = False

    def _icp_single_scale(
        self,
        source: np.ndarray,
        target: np.ndarray,
        target_normals: np.ndarray,
        target_tree: cKDTree,
        init_T: np.ndarray,
        max_dist: float,
        max_iter: int,
    ) -> Tuple[np.ndarray, float, float]:
        """Single-scale Point-to-Plane ICP"""
        T = init_T.copy()
        fitness, rmse = 0.0, float('inf')

        for iteration in range(max_iter):
            # transform source → model coordinate system
            T_inv = np.linalg.inv(T)
            src_model = (T_inv[:3, :3] @ source.T).T + T_inv[:3, 3]

            # nearest neighbor
            dists, indices = target_tree.query(src_model, k=1)
            valid = dists < max_dist
            src_idx = np.where(valid)[0]
            tgt_idx = indices[valid]

            n_corr = len(src_idx)
            if n_corr < 20:
                break

            fitness = n_corr / len(source)
            rmse = np.sqrt(np.mean(dists[valid] ** 2))

            # Point-to-Plane linear system
            s = src_model[src_idx]
            t = target[tgt_idx]
            n = target_normals[tgt_idx]
            diff = s - t

            M = len(src_idx)
            A = np.zeros((M, 6))
            b = np.zeros(M)
            for i in range(M):
                A[i, :3] = np.cross(s[i], n[i])
                A[i, 3:] = n[i]
                b[i] = -np.dot(n[i], diff[i])

            xi, *_ = np.linalg.lstsq(A, b, rcond=None)

            if np.linalg.norm(xi) < self.cfg.icp_tolerance:
                break

            # incremental transformation
            delta = self._twist_to_T(xi)
            T_inv = delta @ T_inv
            T = np.linalg.inv(T_inv)

        return T, fitness, rmse

    def _twist_to_T(self, xi: np.ndarray) -> np.ndarray:
        """Lie algebra → 4×4 transformation matrix"""
        w, t = xi[:3], xi[3:]
        theta = np.linalg.norm(w)
        if theta < 1e-10:
            R = np.eye(3)
        else:
            k = w / theta
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def init_template(
        self, obs_pts_cam: np.ndarray, pose: np.ndarray
    ):
        """First frame: build template (camera → model coordinate system)"""
        T_inv = np.linalg.inv(pose)
        pts_model = (T_inv[:3, :3] @ obs_pts_cam.T).T + T_inv[:3, 3]
        camera_in_model = T_inv[:3, 3]

        pts_model = self.pcu.voxel_downsample(pts_model, self.cfg.icp_voxel_size)
        pts_model = self.pcu.remove_outliers(
            pts_model, self.cfg.outlier_k, self.cfg.outlier_std
        )

        self.template_pts = pts_model
        self.template_normals = self.pcu.estimate_normals(pts_model, ref_point=camera_in_model)
        self.initialized = True
        return len(pts_model)

    def update_template(
        self, obs_pts_cam: np.ndarray, pose: np.ndarray, fitness: float
    ):
        """Incremental template update: fill in occluded areas"""
        if not self.cfg.template_update or fitness < 0.5:
            print(f"Template update disabled or fitness too low: {fitness}")
            return

        T_inv = np.linalg.inv(pose)
        camera_in_model = T_inv[:3, 3]
        new_pts = (T_inv[:3, :3] @ obs_pts_cam.T).T + T_inv[:3, 3]
        new_pts = self.pcu.voxel_downsample(new_pts, self.cfg.icp_voxel_size)

        tree = cKDTree(self.template_pts)
        dists, _ = tree.query(new_pts, k=1)
        novel = dists > self.cfg.template_novel_dist

        if novel.sum() > 0:
            self.template_pts = np.vstack([self.template_pts, new_pts[novel]])
            self.template_pts = self.pcu.voxel_downsample(
                self.template_pts, self.cfg.icp_voxel_size
            )
            self.template_normals = self.pcu.estimate_normals(
                self.template_pts, ref_point=camera_in_model
            )

    def refine(
        self, obs_pts_cam: np.ndarray, init_pose: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Multi-scale ICP refinement"""
        if not self.initialized:
            return init_pose, 0.0, float('inf')

        current_T = init_pose.copy()

        for scale, max_d, max_it in zip(
            self.cfg.icp_scales,
            self.cfg.icp_max_dists,
            self.cfg.icp_max_iters,
        ):
            tgt_down = self.pcu.voxel_downsample(self.template_pts, scale)
            camera_in_model = np.linalg.inv(current_T)[:3, 3]
            tgt_normals = self.pcu.estimate_normals(
                tgt_down, ref_point=camera_in_model
            )
            tgt_tree = cKDTree(tgt_down)
            src_down = self.pcu.voxel_downsample(obs_pts_cam, scale)

            if len(src_down) < 20 or len(tgt_down) < 20:
                continue

            current_T, fitness, rmse = self._icp_single_scale(
                src_down, tgt_down, tgt_normals, tgt_tree,
                current_T, max_d, max_it,
            )

        return current_T, fitness, rmse


# ═══════════════════════════════════════════════════════════════════════════
# 3. PnP Refiner (Feature Matching + solvePnPRansac)
# ═══════════════════════════════════════════════════════════════════════════

class PnPRefiner:
    """
    Pose refinement based on feature matching + PnP.
    Use the features in the first frame YOLO mask + depth backprojection to build a 3D database.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.K = np.array([
            [cfg.fx, 0, cfg.cx],
            [0, cfg.fy, cfg.cy],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist = np.zeros(4, dtype=np.float64)

        # feature extractor
        if cfg.feature_type == "orb":
            self.detector = cv2.ORB_create(nfeatures=cfg.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.detector = cv2.SIFT_create(nfeatures=cfg.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # reference frame database
        self.ref_descriptors = None
        self.ref_points_3d = None
        self.initialized = False

    def _backproject_keypoints(
        self,
        keypoints_2d: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """2D keypoints + depth → 3D (camera coordinate system)"""
        fx, fy = self.cfg.fx, self.cfg.fy
        cx, cy = self.cfg.cx, self.cfg.cy
        h, w = depth.shape

        pts_3d, valid_idx = [], []

        for i, (u, v) in enumerate(keypoints_2d):
            ui, vi = int(round(u)), int(round(v))
            if vi < 0 or vi >= h or ui < 0 or ui >= w:
                continue

            # neighborhood depth median (more robust)
            r = 2
            patch = depth[
                max(0, vi-r):min(h, vi+r+1),
                max(0, ui-r):min(w, ui+r+1),
            ].astype(np.float64)
            patch = patch[patch > 0]

            if len(patch) == 0:
                continue

            z = np.median(patch) / self.cfg.depth_scale
            if z < 0.01 or z > self.cfg.depth_max:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pts_3d.append([x, y, z])
            valid_idx.append(i)

        if len(pts_3d) == 0:
            return np.empty((0, 3)), np.empty(0, dtype=int)
        return np.array(pts_3d), np.array(valid_idx)

    def init_reference(
        self,
        color_bgr: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        pose: np.ndarray,
    ) -> bool:
        """First frame: build feature → 3D database"""
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        mask_cv = PointCloudUtils.prepare_mask(mask, self.cfg.mask_dilate_px)
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask=mask_cv)

        if descriptors is None or len(keypoints) < 10:
            return False

        kp_2d = np.array([kp.pt for kp in keypoints])
        pts_3d_cam, valid_idx = self._backproject_keypoints(kp_2d, depth)

        if len(pts_3d_cam) < 10:
            return False

        # camera → model coordinate system
        T_inv = np.linalg.inv(pose)
        pts_3d_model = (T_inv[:3, :3] @ pts_3d_cam.T).T + T_inv[:3, 3]

        self.ref_descriptors = descriptors[valid_idx]
        self.ref_points_3d = pts_3d_model
        self.initialized = True
        return True

    def refine(
        self,
        color_bgr: np.ndarray,
        mask: np.ndarray,
        init_pose: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Current frame: feature matching → PnP + RANSAC → refine pose

        Returns:
            pnp_pose: [4x4] or None (when failed)
            info: diagnostic information
        """
        info = {
            "success": False,
            "n_matches": 0,
            "n_inliers": 0,
            "inlier_ratio": 0.0,
            "reproj_error": float('inf'),
        }

        if not self.initialized:
            return None, info

        # extract current frame features
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        mask_cv = PointCloudUtils.prepare_mask(mask, self.cfg.mask_dilate_px)
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask=mask_cv)

        if descriptors is None or len(keypoints) < 6:
            return None, info

        # KNN matching + Lowe's ratio test
        matches_knn = self.matcher.knnMatch(
            self.ref_descriptors, descriptors, k=2
        )
        good = []
        for pair in matches_knn:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.cfg.match_ratio * n.distance:
                    good.append(m)

        info["n_matches"] = len(good)
        if len(good) < 6:
            return None, info

        # extract 3D (reference) and 2D (current frame)
        obj_pts = np.array(
            [self.ref_points_3d[m.queryIdx] for m in good], dtype=np.float64
        )
        img_pts = np.array(
            [keypoints[m.trainIdx].pt for m in good], dtype=np.float64
        )

        # solvePnPRansac (EPnP + RANSAC)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=self.K,
            distCoeffs=self.dist,
            iterationsCount=self.cfg.ransac_iter,
            reprojectionError=self.cfg.ransac_reproj,
            confidence=self.cfg.ransac_conf,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success or inliers is None or len(inliers) < 4:
            return None, info

        # LM refinement (using inliers)
        inlier_idx = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=obj_pts[inlier_idx],
            imagePoints=img_pts[inlier_idx],
            cameraMatrix=self.K,
            distCoeffs=self.dist,
            rvec=rvec,
            tvec=tvec,
        )

        # convert to 4×4
        R, _ = cv2.Rodrigues(rvec)
        pnp_pose = np.eye(4)
        pnp_pose[:3, :3] = R
        pnp_pose[:3, 3] = tvec.ravel()

        # reprojection error
        projected, _ = cv2.projectPoints(
            obj_pts[inlier_idx], rvec, tvec, self.K, self.dist
        )
        reproj = np.mean(np.linalg.norm(
            projected.reshape(-1, 2) - img_pts[inlier_idx], axis=1
        ))

        info.update({
            "success": True,
            "n_inliers": len(inliers),
            "inlier_ratio": len(inliers) / len(good),
            "reproj_error": reproj,
        })

        return pnp_pose, info


# ═══════════════════════════════════════════════════════════════════════════
# 4. Temporal Pose Smoother (quaternion weighted average)
# ═══════════════════════════════════════════════════════════════════════════

class PoseSmoother:
    """Temporal Pose Smoothing: exponential decay sliding window"""

    def __init__(self, cfg: PipelineConfig):
        self.window = cfg.smooth_window
        self.decay = cfg.smooth_exp_decay
        self.history: List[np.ndarray] = []

    @staticmethod
    def R_to_quat(R: np.ndarray) -> np.ndarray:
        """rotation matrix → quaternion [w, x, y, z]"""
        tr = np.trace(R)
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            q = np.array([
                0.25 * s,
                (R[2,1] - R[1,2]) / s,
                (R[0,2] - R[2,0]) / s,
                (R[1,0] - R[0,1]) / s,
            ])
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            q = np.array([
                (R[2,1] - R[1,2]) / s,
                0.25 * s,
                (R[0,1] + R[1,0]) / s,
                (R[0,2] + R[2,0]) / s,
            ])
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            q = np.array([
                (R[0,2] - R[2,0]) / s,
                (R[0,1] + R[1,0]) / s,
                0.25 * s,
                (R[1,2] + R[2,1]) / s,
            ])
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            q = np.array([
                (R[1,0] - R[0,1]) / s,
                (R[0,2] + R[2,0]) / s,
                (R[1,2] + R[2,1]) / s,
                0.25 * s,
            ])
        return q / np.linalg.norm(q)

    @staticmethod
    def quat_to_R(q: np.ndarray) -> np.ndarray:
        """quaternion → rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
        ])

    def smooth(self, pose: np.ndarray) -> np.ndarray:
        """input current frame pose, return smoothed pose"""
        self.history.append(pose.copy())
        if len(self.history) > self.window:
            self.history.pop(0)

        n = len(self.history)
        if n == 1:
            return pose

        # exponential decay weights
        w = np.exp(np.arange(n) * self.decay)
        w /= w.sum()

        # translation: weighted average
        t_smooth = sum(
            wi * h[:3, 3] for wi, h in zip(w, self.history)
        )

        # rotation: quaternion weighted average
        quats = [self.R_to_quat(h[:3, :3]) for h in self.history]
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[0]) < 0:
                quats[i] = -quats[i]

        q_avg = sum(wi * q for wi, q in zip(w, quats))
        q_avg /= np.linalg.norm(q_avg)

        out = np.eye(4)
        out[:3, :3] = self.quat_to_R(q_avg)
        out[:3, 3] = t_smooth
        return out

    def reset(self):
        self.history.clear()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Pose Evaluator (ADD / ADD-S / 5cm5° / Reprojection)
# ═══════════════════════════════════════════════════════════════════════════

class PoseEvaluator:
    """6D Pose 精度评估 (可选模块, 需要 GT)"""

    def __init__(self, K: np.ndarray):
        self.K = K

    def add_s(
        self, pts: np.ndarray,
        R_pred: np.ndarray, t_pred: np.ndarray,
        R_gt: np.ndarray, t_gt: np.ndarray,
    ) -> float:
        """ADD-S (对称物体)"""
        pred = (R_pred @ pts.T).T + t_pred
        gt = (R_gt @ pts.T).T + t_gt
        tree = cKDTree(gt)
        dists, _ = tree.query(pred, k=1)
        return np.mean(dists)

    def rotation_error(self, R_pred, R_gt) -> float:
        """旋转误差 (度)"""
        tr = np.clip(np.trace(R_pred.T @ R_gt), -1.0, 3.0)
        return np.degrees(np.arccos((tr - 1.0) / 2.0))

    def translation_error(self, t_pred, t_gt) -> float:
        """平移误差 (m)"""
        return np.linalg.norm(t_pred - t_gt)

    def reproj_error(
        self, pts, R_pred, t_pred, R_gt, t_gt
    ) -> float:
        """2D 重投影误差 (px)"""
        def proj(R, t):
            p3d = (R @ pts.T).T + t
            p2d = (self.K @ p3d.T).T
            return p2d[:, :2] / p2d[:, 2:3]
        return np.mean(np.linalg.norm(proj(R_pred, t_pred) - proj(R_gt, t_gt), axis=1))

    def evaluate(
        self, pts, R_pred, t_pred, R_gt, t_gt
    ) -> Dict[str, float]:
        return {
            "ADD-S (m)": self.add_s(pts, R_pred, t_pred, R_gt, t_gt),
            "Rotation (°)": self.rotation_error(R_pred, R_gt),
            "Translation (m)": self.translation_error(t_pred, t_gt),
            "Reproj (px)": self.reproj_error(pts, R_pred, t_pred, R_gt, t_gt),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Pose Fusion (PnP + ICP 加权融合)
# ═══════════════════════════════════════════════════════════════════════════

class PoseFusion:
    """融合 PnP (旋转强) 和 ICP (平移强) 的结果"""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    @staticmethod
    def _reproj_error(
        obj_pts, img_pts, pose, K, dist
    ) -> float:
        rvec, _ = cv2.Rodrigues(pose[:3, :3])
        proj, _ = cv2.projectPoints(
            obj_pts, rvec, pose[:3, 3], K, dist
        )
        return np.mean(np.linalg.norm(
            proj.reshape(-1, 2) - img_pts, axis=1
        ))

    def fuse(
        self,
        densefusion_pose: np.ndarray,
        icp_pose: np.ndarray,
        icp_fitness: float,
        pnp_pose: Optional[np.ndarray],
        pnp_info: dict,
    ) -> Tuple[np.ndarray, str]:
        """
        融合策略:
          - 平移: 优先 ICP (几何对齐更准)
          - 旋转: PnP 成功且 inlier 高时用 PnP, 否则用 ICP
          - ICP 失败时回退 DenseFusion

        Returns:
            fused_pose [4x4], chosen_rotation ("pnp" / "icp" / "densefusion")
        """
        # 回退判断
        if icp_fitness < self.cfg.icp_fallback_fitness:
            return densefusion_pose, "densefusion"

        # 平移: 始终用 ICP
        t_fused = icp_pose[:3, 3]

        # 旋转: PnP 优先 (如果成功且质量好)
        chosen_rot = "icp"
        R_fused = icp_pose[:3, :3]

        if (pnp_pose is not None
            and pnp_info.get("success", False)
            and pnp_info.get("inlier_ratio", 0) > 0.3
            and pnp_info.get("reproj_error", float('inf')) < 5.0):

            chosen_rot = "pnp"
            R_fused = pnp_pose[:3, :3]

        fused = np.eye(4)
        fused[:3, :3] = R_fused
        fused[:3, 3] = t_fused
        return fused, chosen_rot


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main Pipeline (整合所有模块)
# ═══════════════════════════════════════════════════════════════════════════

class PoseRefinementPipeline:
    """
    ┌──────────────┐
    │  DenseFusion  │  初始 Pose
    └──────┬───────┘
           ▼
    ┌──────────────┐   ┌──────────────┐
    │ Template ICP  │   │  PnP Refiner  │   并行
    │  (平移精化)   │   │  (旋转精化)   │
    └──────┬───────┘   └──────┬───────┘
           ▼                  ▼
    ┌──────────────────────────────┐
    │        Pose Fusion           │
    │  ICP 平移 + PnP/ICP 旋转     │
    └──────────────┬───────────────┘
                   ▼
    ┌──────────────────────────────┐
    │     Temporal Smoothing       │
    └──────────────┬───────────────┘
                   ▼
              Final Pose
    """

    def __init__(self, cfg: PipelineConfig = None):
        self.cfg = cfg or PipelineConfig()
        self.pcu = PointCloudUtils()

        # 子模块
        self.icp_refiner = TemplateICPRefiner(self.cfg)
        self.pnp_refiner = PnPRefiner(self.cfg)
        self.fusion = PoseFusion(self.cfg)
        self.smoother = PoseSmoother(self.cfg)
        self.evaluator = PoseEvaluator(np.array([
            [self.cfg.fx, 0, self.cfg.cx],
            [0, self.cfg.fy, self.cfg.cy],
            [0, 0, 1],
        ]))

        self.frame_count = 0

    def process_frame(
        self,
        color_bgr: np.ndarray,        # [H, W, 3] BGR
        depth: np.ndarray,             # [H, W] uint16 (mm)
        yolo_mask: np.ndarray,         # [H, W] bool
        densefusion_pose: np.ndarray,  # [4x4]
        gt_pose: np.ndarray = None,    # [4x4] (optional, for evaluation)
        verbose: bool = False,
    ) -> RefinementResult:
        """
        处理单帧。

        Args:
            color_bgr:        Orbbec D335 BGR 图像
            depth:            Orbbec D335 深度图 (uint16, mm)
            yolo_mask:        YOLO11 Seg 输出的物体 mask
            densefusion_pose: DenseFusion 初始 pose [4x4]
            gt_pose:          Ground Truth (可选)
            verbose:          打印详细信息

        Returns:
            RefinementResult
        """
        self.frame_count += 1
        result = RefinementResult(
            pose=densefusion_pose.copy(),
            frame_id=self.frame_count,
        )

        # ── 提取点云 ──
        obs_pts = self.pcu.rgbd_to_points(
            depth, yolo_mask,
            self.cfg.fx, self.cfg.fy, self.cfg.cx, self.cfg.cy,
            self.cfg.depth_scale, self.cfg.depth_max,
        )

        if len(obs_pts) < 50:
            result.status = "insufficient_points"
            return result

        obs_pts = self.pcu.remove_outliers(
            obs_pts, self.cfg.outlier_k, self.cfg.outlier_std
        )

        # ════════════════════════════════════════
        # 首帧: 初始化所有模块
        # ════════════════════════════════════════
        if self.frame_count == 1:
            n_template = self.icp_refiner.init_template(obs_pts, densefusion_pose)
            pnp_ok = self.pnp_refiner.init_reference(
                color_bgr, depth, yolo_mask, densefusion_pose
            )

            result.status = "initialized"
            if verbose:
                print(f"[Frame 1] Template: {n_template} pts | "
                      f"PnP ref: {'OK' if pnp_ok else 'FAIL'} "
                      f"({len(self.pnp_refiner.ref_points_3d) if pnp_ok else 0} features)")
            return result

        # ════════════════════════════════════════
        # 后续帧: 精化
        # ════════════════════════════════════════

        # ── ICP 精化 ──
        icp_pose, icp_fitness, icp_rmse = self.icp_refiner.refine(
            obs_pts, densefusion_pose
        )
        result.icp_fitness = icp_fitness
        result.icp_rmse = icp_rmse

        # ── PnP 精化 ──
        pnp_pose, pnp_info = self.pnp_refiner.refine(
            color_bgr, yolo_mask, densefusion_pose
        )
        result.pnp_success = pnp_info.get("success", False)
        result.pnp_inliers = pnp_info.get("n_inliers", 0)
        result.pnp_reproj_error = pnp_info.get("reproj_error", float('inf'))

        # ── 融合 ──
        fused_pose, chosen_rot = self.fusion.fuse(
            densefusion_pose, icp_pose, icp_fitness,
            pnp_pose, pnp_info,
        )
        result.chosen_rotation = chosen_rot

        # ── 时序平滑 ──
        smoothed_pose = self.smoother.smooth(fused_pose)

        # ── 模板更新 ──
        self.icp_refiner.update_template(obs_pts, smoothed_pose, icp_fitness)

        # ── 最终结果 ──
        result.pose = smoothed_pose
        result.R = smoothed_pose[:3, :3]
        result.t = smoothed_pose[:3, 3]
        result.status = "refined"

        # ── 评估 (如果有 GT) ──
        if gt_pose is not None:
            template_pts = self.icp_refiner.template_pts
            if template_pts is not None:
                metrics_before = self.evaluator.evaluate(
                    template_pts,
                    densefusion_pose[:3, :3], densefusion_pose[:3, 3],
                    gt_pose[:3, :3], gt_pose[:3, 3],
                )
                metrics_after = self.evaluator.evaluate(
                    template_pts,
                    result.R, result.t,
                    gt_pose[:3, :3], gt_pose[:3, 3],
                )

                if verbose:
                    print(f"  [Eval] Before → After:")
                    for k in metrics_before:
                        b, a = metrics_before[k], metrics_after[k]
                        imp = (b - a) / b * 100 if b > 0 else 0
                        print(f"    {k}: {b:.4f} → {a:.4f} ({imp:+.1f}%)")

        if verbose:
            print(
                f"[Frame {self.frame_count:3d}] "
                f"ICP: fit={icp_fitness:.3f} rmse={icp_rmse:.5f} | "
                f"PnP: {'✓' if result.pnp_success else '✗'} "
                f"inl={result.pnp_inliers} "
                f"reproj={result.pnp_reproj_error:.2f}px | "
                f"Rot={chosen_rot}"
            )

        return result

    def reset(self):
        """重置所有状态 (物体丢失/重新检测时调用)"""
        self.icp_refiner = TemplateICPRefiner(self.cfg)
        self.pnp_refiner = PnPRefiner(self.cfg)
        self.smoother.reset()
        self.frame_count = 0


# ═══════════════════════════════════════════════════════════════════════════
# 8. Usage Example
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 配置 (根据你的 Orbbec D335 调整) ──
    cfg = PipelineConfig(
        fx=605.286, fy=605.699,
        cx=323.799, cy=248.846,
        depth_scale=1000.0,
        depth_max=1.0,
    )