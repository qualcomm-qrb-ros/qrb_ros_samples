# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
# SPDX-License-Identifier: BSD-3-Clause-Clear

from setuptools import find_packages, setup

package_name = 'simulation_sample_amr_simple_motion'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Weijie Shen',
    maintainer_email='weijshen@qti.qualcomm.com',
    description='package for simple motion control of QRB AMRs within the simulator',
    license='BSD-3-Clause-Clear',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_motion = simulation_sample_amr_simple_motion.simple_motion:main',
        ],
    },
)
