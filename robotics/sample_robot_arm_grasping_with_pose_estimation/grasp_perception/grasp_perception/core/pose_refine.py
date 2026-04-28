import numpy as np
from scipy.spatial import cKDTree


class TemplateICPRefiner:
    """
    No-CAD, No-Geometry-Constraint Pose Refinement
    Only: RGB-D + YOLO mask + Point-to-Plane ICP
    """

    def __init__(
        self,
        voxel_size=0.002,
        max_dist=0.50,
        max_iter=50,
        tol=1e-6,
    ):
        self.voxel = voxel_size
        self.max_dist = max_dist
        self.max_iter = max_iter
        self.tol = tol

        self.template_pts = None
        self.template_normals = None
        self.initialized = False

    # ---------- utils ----------

    def voxel_down(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            pts = pts.reshape(-1, 3)

        idx = np.floor(pts / self.voxel).astype(np.int64)
        vox = {}
        for p, i in zip(pts, idx):
            key = tuple(i.tolist())
            vox.setdefault(key, []).append(p)
        return np.array([np.mean(v, axis=0) for v in vox.values()])

    def estimate_normals(self, pts, k=20):
        tree = cKDTree(pts)
        _, idx = tree.query(pts, k=k)
        normals = np.zeros_like(pts)
        for i in range(len(pts)):
            nbr = pts[idx[i]]
            cov = np.cov(nbr.T)
            _, vec = np.linalg.eigh(cov)
            normals[i] = vec[:, 0]
        normals *= np.sign(np.sum(normals * (-pts), axis=1))[:, None]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return normals

    # ---------- ICP ----------

    def point_to_plane_icp(self, src, tgt, tgt_n, init_T):
        T = init_T.copy()
        tree = cKDTree(tgt)

        for _ in range(self.max_iter):
            src_tf = (T[:3, :3] @ src.T).T + T[:3, 3]
            d, idx = tree.query(src_tf)
            m = d < self.max_dist
            if m.sum() < 20:
                break

            A, b = [], []
            for s, t, n in zip(src_tf[m], tgt[idx[m]], tgt_n[idx[m]]):
                A.append(np.r_[np.cross(s, n), n])
                b.append(-np.dot(n, s - t))

            xi, *_ = np.linalg.lstsq(np.vstack(A), np.array(b), rcond=None)

            if np.linalg.norm(xi) < self.tol:
                break

            w, t = xi[:3], xi[3:]
            θ = np.linalg.norm(w)
            if θ > 1e-9:
                k = w / θ
                K = np.array([[0, -k[2], k[1]],
                              [k[2], 0, -k[0]],
                              [-k[1], k[0], 0]])
                dR = np.eye(3) + np.sin(θ)*K + (1-np.cos(θ))*(K@K)
            else:
                dR = np.eye(3)

            dT = np.eye(4)
            dT[:3, :3] = dR
            dT[:3, 3] = t
            T = dT @ T

        return T

    # ---------- main ----------

    def _to_transform(self, init_pose):
        pose = np.asarray(init_pose, dtype=np.float64)
        if pose.shape == (4, 4):
            return pose.copy()
        if pose.shape == (3,):
            T = np.eye(4, dtype=np.float64)
            T[:3, 3] = pose
            return T
        raise ValueError(f"init_pose must be 4x4 matrix or translation(3,), got {pose.shape}")

    def refine(self, obs_pts, init_pose):
        obs_pts = self.voxel_down(obs_pts)
        init_T = self._to_transform(init_pose)
        if not np.all(np.isfinite(init_T)):
            raise ValueError("init_pose contains non-finite values")

        if len(obs_pts) < 20:
            return init_T[:3, 3]

        if not self.initialized:
            # Initialization: build template (camera → template)
            Tinv = np.linalg.inv(init_T)
            self.template_pts = self.voxel_down(
                (Tinv[:3, :3] @ obs_pts.T).T + Tinv[:3, 3]
            )
            if len(self.template_pts) < 20:
                return init_T[:3, 3]
            self.template_normals = self.estimate_normals(self.template_pts)
            self.initialized = True
            return init_T[:3, 3]

        refined_T = self.point_to_plane_icp(
            obs_pts, self.template_pts, self.template_normals, init_T
        )
        return refined_T[:3, 3]