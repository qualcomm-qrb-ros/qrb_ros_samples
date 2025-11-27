# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'sample_depth_estimation'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(where='.', exclude=['test']),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
                 ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob(os.path.join('resource', 'input_image.jpg'))),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
        (os.path.join('lib', package_name, package_name), glob(os.path.join(package_name, '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Teng Fan',
    maintainer_email='quic_tengf@quicinc.com',
    description='Deep Convolutional Neural Network model for depth estimation',
    license='BSD-3-Clause-Clear',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_estimation_node = sample_depth_estimation.depth_estimation_node:main',
        ],
    },
)
