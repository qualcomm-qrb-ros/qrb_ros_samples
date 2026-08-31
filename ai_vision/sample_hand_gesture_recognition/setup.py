# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from setuptools import find_packages, setup

package_name = 'sample_hand_gesture_recognition'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['resource/input_image.jpg']),
        ('lib/' + package_name, [package_name + '/gesture_node.py']),
        ('lib/' + package_name, [package_name + '/gesture_fp32.py']),
        ('lib/' + package_name, [package_name + '/gesture_classifier.py']),
        ('lib/' + package_name, [package_name + '/mediapipe_hand_base.py']),
        ('lib/' + package_name, [package_name + '/visualization.py']),
        ('share/' + package_name, ['launch/launch_with_image_publisher.py']),
        ('share/' + package_name, ['launch/launch_with_qrb_ros_camera.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yi Li',
    maintainer_email='quic_yli7@quicinc.com',
    description='sample_hand_gesture_recognition is a Python-based hand gesture '
                'recognition ROS node. It runs the MediaPipe Hand palm and '
                'landmark detectors on the NPU via QNN and classifies the hand '
                'gesture into 8 classes in-node with a lightweight FP32 model.',
    license='BSD-3-Clause-Clear',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qrb_ros_gesture_recognition = sample_hand_gesture_recognition.gesture_node:main'
        ],
    },
)
