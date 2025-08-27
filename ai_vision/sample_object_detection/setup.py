# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from setuptools import find_packages, setup
from glob import glob

package_name = 'sample_object_detection'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name + '/launch',  ['launch/launch_with_orbbec_camera.py']),
        ('share/' + package_name + '/launch',  ['launch/launch_with_qrb_ros_camera.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fulan Liu',
    maintainer_email='fulaliu@qti.qualcomm.com',
    description='sample object detection with yolov8 model',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
