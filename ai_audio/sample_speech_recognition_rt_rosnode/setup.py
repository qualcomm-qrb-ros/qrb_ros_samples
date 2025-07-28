# Copyright (c) 2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from setuptools import setup, find_packages

package_name = 'qrb_ros_speech_recognition'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[
        'audio_process',
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
            'plugin = SpeechRecognition.plugin:main',
        ],
    },
)
