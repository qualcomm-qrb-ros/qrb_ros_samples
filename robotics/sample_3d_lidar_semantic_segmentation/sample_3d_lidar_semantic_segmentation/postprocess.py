# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


def get_color_map():
    """RGB color mapping for 20 classes"""
    cmap = np.zeros((20, 3), dtype=np.uint32)
    cmap[0] = [0, 0, 0];          cmap[1] = [245, 150, 100];   cmap[2] = [245, 230, 100]
    cmap[3] = [150, 60, 30];      cmap[4] = [180, 30, 80];     cmap[5] = [255, 0, 0]
    cmap[6] = [30, 30, 255];      cmap[7] = [200, 40, 255];    cmap[8] = [90, 30, 150]
    cmap[9] = [255, 0, 255];      cmap[10] = [255, 150, 255];  cmap[11] = [75, 0, 75]
    cmap[12] = [75, 0, 175];      cmap[13] = [0, 200, 255];    cmap[14] = [50, 120, 255]
    cmap[15] = [0, 175, 0];       cmap[16] = [0, 60, 135];     cmap[17] = [80, 240, 150]
    cmap[18] = [150, 240, 255];   cmap[19] = [0, 0, 255]
    return cmap

def postprocess(original_pointcloud_msg, output_tensor, point_indices, num_points, original_points_data):
    """
    Postprocess SalsaNext model output to create semantic point cloud.
    
    Args:
        original_pointcloud_msg: Original PointCloud2 message (for header)
        output_tensor: Model output tensor [1, 20, 64, 2048]
        point_indices: Tuple of (proj_y, proj_x) indices from preprocess
        num_points: Number of points in the cloud
        original_points_data: Tuple of (x, y, z, intensity) from preprocess

    Returns:
        semantic_pointcloud_msg: PointCloud2 message with semantic colors
    """
    if num_points == 0:
        return None
    
    proj_y, proj_x = point_indices
    x, y, z, intensity = original_points_data

    # Get color map
    color_map = get_color_map()

    # ---- Post-processing and color mapping ----
    predictions = np.argmax(output_tensor[0], axis=0)  # [64, 2048]
    point_labels = predictions[proj_y, proj_x]         # Extract label for each 3D point
    point_colors = color_map[point_labels]
    
    # Pack RGB into a single uint32 value (0x00RRGGBB layout)
    r, g, b = point_colors[:, 0], point_colors[:, 1], point_colors[:, 2]
    rgb_packed = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
    
    # Create semantic point cloud structure
    semantic_points = np.zeros(num_points, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), 
        ('intensity', np.float32), ('rgb', np.uint32)
    ])
    
    semantic_points['x'] = x
    semantic_points['y'] = y
    semantic_points['z'] = z
    semantic_points['intensity'] = intensity
    semantic_points['rgb'] = rgb_packed
    
    # Create PointCloud2 message
    msg_out = PointCloud2()
    msg_out.header = original_pointcloud_msg.header
    msg_out.height = 1
    msg_out.width = num_points
    msg_out.is_dense = False
    msg_out.is_bigendian = False
    
    msg_out.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        # RViz expects the rgb field to be declared as FLOAT32 (datatype=7)
        # even though the underlying bytes are a packed uint32 (0x00RRGGBB).
        # Using UINT32 here causes RViz to ignore the colour channel.
        PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1),
    ]
    msg_out.point_step = 20
    msg_out.row_step = msg_out.point_step * msg_out.width
    msg_out.data = semantic_points.tobytes()
    
    return msg_out