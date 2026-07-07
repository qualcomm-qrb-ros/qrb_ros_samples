# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from glob import glob
from setuptools import setup


package_name = 'sample_midas_yolo_parallel'


setup(
    name=package_name,
    version='1.0.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Qualcomm Technologies, Inc.',
    maintainer_email='opensource@qualcomm.com',
    description='Parallel MiDaS and YOLO segmentation pipeline using ROS nodes and QNN inference',
    license='BSD-3-Clause-Clear',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
