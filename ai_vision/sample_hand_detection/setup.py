# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from setuptools import find_packages, setup

package_name = 'sample_hand_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['resource/' +"/input_image.jpg"]),
        ('lib/' + package_name, [package_name + "/hand_landmark_detector.py"]),
        ('lib/' + package_name, [package_name + "/hand_plam_detector.py"]),
        ('lib/' + package_name, [package_name + "/visualization.py"]),
        ('share/' + package_name, ['launch/' + "/launch_with_usbcam.py"]),
        ('share/' + package_name, ['launch/' + "/launch_with_image_publisher.py"]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dapeng Yuan',
    maintainer_email='quic_dapeyuan@quicinc.com',
    description='sample_hand_detection is a Python-based hand recognition ROS node that uses QNN for model inference.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qrb_ros_hand_detector = sample_hand_detection.hand_landmark_detector:main'
        ],
    },
)
