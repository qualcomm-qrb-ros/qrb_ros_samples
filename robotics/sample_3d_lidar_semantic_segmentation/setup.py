# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sample_3d_lidar_semantic_segmentation'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.py')),
        ('lib/' + package_name, [package_name + "/lidar_semantic_segmentation.py"]),
        ('lib/' + package_name, [package_name + "/preprocess.py"]),
        ('lib/' + package_name, [package_name + "/postprocess.py"]),
        ('lib/' + package_name, [package_name + "/__init__.py"]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='guohmiao',
    maintainer_email='guohmiao@qti.qualcomm.com',
    description='3D LiDAR Semantic Segmentation using SalsaNext model with QNN inference',
    license='BSD-3-Clause-Clear',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lidar_semantic_segmentation_node = sample_3d_lidar_semantic_segmentation.lidar_semantic_segmentation:main'
        ],
    },
)