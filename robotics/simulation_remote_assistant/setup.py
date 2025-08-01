# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
# SPDX-License-Identifier: BSD-3-Clause-Clear

from setuptools import setup

package_name = 'simulation_remote_assistant'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/locations.yaml']),
        ('share/' + package_name + '/config', ['config/objects.yaml']),
        ('share/' + package_name + '/scripts', ['scripts/map_nav_setup.py']),
        ('share/' + package_name + '/launch', ['launch/yolo_detectcion.launch.py']),
        ('share/' + package_name + '/launch', ['launch/map_nav_setup.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xionfu',
    maintainer_email='xionfu@qti.qualcomm.com',
    description='package for remote assistant simluation',
    license='BSD-3-Clause-Clear',
    entry_points={
        'console_scripts': [
            'build_map_node = simulation_remote_assistant.build_map_node:main',
            'nav_preparation_node = simulation_remote_assistant.nav_preparation_node:main',
            'task_manager_node = simulation_remote_assistant.task_manager_node:main',
        ],
    },
)
