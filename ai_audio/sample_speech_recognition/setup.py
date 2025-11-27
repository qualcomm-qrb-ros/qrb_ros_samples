# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from setuptools import find_packages, setup

package_name = 'sample_speech_recognition'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name,[package_name + "/audio_process.py"]),
        ('lib/' + package_name,[package_name + "/qrb_ros_whisper.py"]),
        ('lib/' + package_name,[package_name + "/decode.py"]),
	('share/' + package_name, ['resource/' +"/decode_input.txt"]),
	('share/' + package_name, ['resource/' +"/encode_input.txt"]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fulan Liu',
    maintainer_email='quic_fulaliu@quicinc.com',
    description='speech recognition with whisper model',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qrb_ros_whisper = sample_speech_recognition.qrb_ros_whisper:main'
        ],
    },
)
