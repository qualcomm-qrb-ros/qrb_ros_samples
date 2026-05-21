# Copyright (c) 2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'qrb_ros_speech_recognition_rt_rosnode'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[
        'audio_process',
        'qrb_ros_android_asr',
        'qrb_ros_speech_recognition',
        'tflite_load_model',
        'whisper_transcription'
    ],
    install_requires=['setuptools'],
    package_data={
        '': ['*.py'],
    },
    include_package_data=True,
    zip_safe=True,
    maintainer='shouyi hu',
    maintainer_email='quic_shouhu@quicinc.com',
    description='QRB ROS Speech Recognition',
    license='BSD-3-Clause-Clear',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qrb_ros_speech_recognition = qrb_ros_speech_recognition:main',
            'qrb_ros_android_asr = qrb_ros_android_asr:main',
        ],
    },
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
)