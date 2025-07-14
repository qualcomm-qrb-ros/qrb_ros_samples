'''
Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear
'''
import os
from setuptools import setup
from setuptools.command.install import install

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        files = ("ocr_server","ocr_client","ocr_testnode","ocr_service")
        for index in files:
            file_path = os.path.join(self.install_lib,"ocr_service","../../../ocr_service/"+index)
            with open(file_path,"r") as file:
                lines = file.readlines()
                lines[0] = '#!/usr/bin/python \n'
            with open(file_path, 'w') as file:
                file.writelines(lines)

package_name = 'ocr_service'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chalia',
    maintainer_email='quic_chalia@quicinc.com',
    description='ocr service and publish result',
    license='BSD-3-Clause-Clear',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ocr_service = ocr_service.ocr_service:main',
            'ocr_server = ocr_service.ocr_server:main',
            'ocr_client = ocr_service.ocr_client:main',
            'ocr_testnode = ocr_service.ocr_testnode:main'
        ],
    },
    cmdclass={
    'install':CustomInstallCommand
    }
)
