import os
import numpy as np
import cv2
from ament_index_python.packages import get_package_share_directory

package_share_directory = get_package_share_directory("sample_hrnet_pose_estimation")

def preprocess(image):

    input_image = cv2.resize(image, (256, 192), interpolation = cv2.INTER_AREA)
    input_image = input_image.reshape((1,192,256,3))
    input_image = input_image.transpose((0, 2, 1, 3))
    input_image = input_image.astype(np.float32, copy=False) / 255.0
    
    return input_image


