# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from setuptools import setup
import os
from glob import glob

package_name = 'sample_fall_detection'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fulan Liu',
    maintainer_email='fulaliu@qti.qualcomm.com',
    description='Sample fall detection node with Edge Impulse inference and QRB ROS camera launch.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fall_state_gate_node = sample_fall_detection.fall_detect_edge_impulse:main',
        ],
    },
)
