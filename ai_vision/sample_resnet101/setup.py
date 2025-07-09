from setuptools import find_packages, setup
from glob import glob

package_name = 'sample_resnet101_quantized'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	('share/' + package_name, glob('resource/*.jpg')),
        ('lib/' + package_name,[package_name + "/qrb_ros_resnet101.py"]),
        ('lib/' + package_name,[package_name + "/qrb_ros_resnet101_posprocess.py"]),
	('share/' + package_name, ['resource/' +"/input.txt"]),
        ('share/' + package_name + '/launch', ['launch/launch_with_usbcam.py']),
	('share/' + package_name + '/launch',  ['launch/launch_with_image_publisher.py']),
        ('share/' + package_name + '/launch',  ['launch/launch_with_orbbec_camera.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fulan Liu',
    maintainer_email='quic_fulaliu@quicinc.com',
    description='Imagenet classifier and general purpose backbone with resnet101 model',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qrb_ros_resnet101 = sample_resnet101_quantized.qrb_ros_resnet101:main',
            'qrb_ros_resnet101_posprocess  = sample_resnet101_quantized.qrb_ros_resnet101_posprocess:main'
        ],
    },
)
