# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from setuptools import find_packages, setup

package_name = 'sample_hrnet_pose_estimation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['resource/' +"/input_image.jpg"]),
        ('lib/' + package_name, [package_name + "/hrnet_pose_estimation.py"]),
        ('lib/' + package_name, [package_name + "/preprocess.py"]),
        ('lib/' + package_name, [package_name + "/postprocess.py"]),
        ('share/' + package_name, ['launch/' + "/launch_with_usbcam.py"]),
        ('share/' + package_name, ['launch/' + "/launch_with_image_publisher.py"]),
        ('share/' + package_name, ['launch/' + "/launch_with_qrb_ros_camera.py"])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='guohmiao',
    maintainer_email='guohmiao@qti.qualcomm.com',
    description='sample_hrnet_pose_estimation is a Python-based human pose estimation ROS node that uses QNN for model inference.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sample_hrnet_pose_estimation = sample_hrnet_pose_estimation.hrnet_pose_estimation:main'
        ],
    },
)
