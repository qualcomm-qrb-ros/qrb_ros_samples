# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sample_face_detection'

setup(
    name=package_name,
    version='1.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name), glob(os.path.join('resource', 'face_image.jpg'))),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
        (os.path.join('lib', package_name, package_name), glob(os.path.join(package_name, '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chuhao Xie',
    maintainer_email='chuhxie@qti.qualcomm.com',
    description='sample_face_detection is a Python-based face recognition ROS node that uses QNN for model inference.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'qrb_ros_face_detector = sample_face_detection.mediapipe_face_node:main',
        ],
    },
)
