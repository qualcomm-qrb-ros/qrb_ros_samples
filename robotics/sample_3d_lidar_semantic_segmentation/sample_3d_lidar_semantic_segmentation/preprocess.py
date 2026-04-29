# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2


def preprocess(pointcloud_msg, fov_up=3.0, fov_down=-25.0):
    """
    Preprocess point cloud data for SalsaNext model inference.
    
    Args:
        pointcloud_msg: PointCloud2 message
        fov_up: Vertical field of view upper limit (degrees)
        fov_down: Vertical field of view lower limit (degrees)
    
    Returns:
        input_tensor: Processed projection image [1, 64, 2048, 5]
        point_indices: Tuple of (proj_y, proj_x) for mapping back to 3D points
        num_points: Number of points in the cloud
        original_points: Tuple of (x, y, z, intensity) original point data
    """
    # Convert degrees to radians
    fov_up_rad = fov_up / 180.0 * np.pi
    fov_down_rad = fov_down / 180.0 * np.pi
    fov = fov_up_rad - fov_down_rad
    
    # Projection parameters
    proj_H = 64
    proj_W = 2048
    
    # SemanticKITTI original mean and std: [depth, x, y, z, intensity]
    sensor_img_means = np.array([12.12, 10.88, 0.23, -1.04, 0.21], dtype=np.float32)
    sensor_img_stds = np.array([12.32, 11.47, 6.91, 0.86, 0.16], dtype=np.float32)
    
    # 1. Parse point cloud data
    pc_data = pc2.read_points(pointcloud_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    
    # Convert to structured array
    struct_array = np.array(list(pc_data))
    
    num_points = struct_array.shape[0]
    if num_points == 0:
        return None, None, 0, None
    
    # Extract column data safely
    x = struct_array['x'].astype(np.float32)
    y = struct_array['y'].astype(np.float32)
    z = struct_array['z'].astype(np.float32)
    intensity = struct_array['intensity'].astype(np.float32)
    
    # ---- Spherical projection logic ----
    # Combine x, y, z into [N, 3] matrix to compute depth
    xyz = np.column_stack((x, y, z))
    depth = np.linalg.norm(xyz, axis=1)
    depth = np.maximum(depth, 1e-5)  # Prevent division by zero
    
    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(z / depth)
    
    # Calculate projection coordinates
    proj_x = 0.5 * (yaw / np.pi + 1.0) * proj_W
    proj_y = (1.0 - (pitch + abs(fov_down_rad)) / fov) * proj_H
    
    # Convert to integer coordinates
    proj_x = np.floor(proj_x).astype(np.int32)
    proj_y = np.floor(proj_y).astype(np.int32)
    
    # Clip boundaries
    proj_x = np.clip(proj_x, 0, proj_W - 1)
    proj_y = np.clip(proj_y, 0, proj_H - 1)
    
    # ---- Depth sorting to solve occlusion problem ----
    # Far points are written first, near points overwrite later
    order = np.argsort(depth)[::-1]
    proj_y_sorted, proj_x_sorted = proj_y[order], proj_x[order]
    
    # Initialize projection image and mask
    proj_img = np.zeros((proj_H, proj_W, 5), dtype=np.float32)
    proj_mask = np.zeros((proj_H, proj_W), dtype=np.float32)
    
    # Fill projection image
    proj_img[proj_y_sorted, proj_x_sorted, 0] = depth[order]
    proj_img[proj_y_sorted, proj_x_sorted, 1] = x[order]
    proj_img[proj_y_sorted, proj_x_sorted, 2] = y[order]
    proj_img[proj_y_sorted, proj_x_sorted, 3] = z[order]
    proj_img[proj_y_sorted, proj_x_sorted, 4] = intensity[order]
    
    # Record positions with laser points
    proj_mask[proj_y_sorted, proj_x_sorted] = 1.0
    
    # ---- Data normalization ----
    proj_img = (proj_img - sensor_img_means) / sensor_img_stds
    # Reset blank areas without points to 0
    proj_img = proj_img * proj_mask[:, :, np.newaxis]
    
    # Expand dimensions for batch
    input_tensor = np.expand_dims(proj_img, axis=0)  # [1, 64, 2048, 5]

    return input_tensor, (proj_y, proj_x), num_points, (x, y, z, intensity)