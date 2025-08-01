import numpy as np
import cv2
import os

from ament_index_python.packages import get_package_share_directory

package_share_directory = get_package_share_directory("sample_hrnet_pose_estimation")

def postprocess(image,output_tensor):
    height, width, _ = image.shape
    output_data = output_tensor.reshape((64, 48, 17))
    
    max_confidence_index = []
    max_confidence_value = []
    for z in range(output_data.shape[2]):
        max_index = np.unravel_index(np.argmax(output_data[:,:,z]), output_data[:,:,z].shape)
        max_value = np.max(output_data[:, :, z])
        max_confidence_value.append(max_value)
        max_confidence_index.append(max_index)

    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6],
        [5, 7], [6, 8],
        [7, 9], [8, 10],
        [1, 2],
        [0, 1], [0, 2],
        [1, 3], [2, 4],
        [3, 5], [4, 6]
    ]

    center_points = []

    the_minimum_confidence = 0.3

    for z,points in enumerate(max_confidence_index):
        if max_confidence_value[z] > the_minimum_confidence:
            x, y = points
            center_x = int(x / 64 * width)
            center_y = int(y / 48 * height)
            cv2.circle(image, (center_x, center_y), 6, (0, 0, 255), -1)
            center_points.append((center_x, center_y))

    for connection in skeleton:
        if max_confidence_value[connection[0]] > the_minimum_confidence and max_confidence_value[connection[1]] > the_minimum_confidence:
            start_point = max_confidence_index[connection[0]]
            end_point = max_confidence_index[connection[1]]
            start_x, start_y = int(start_point[0] / 64 * width), int(start_point[1] / 48 * height)
            end_x, end_y = int(end_point[0] / 64 * width), int(end_point[1] / 48 * height)
            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 4)

    return image,center_points
