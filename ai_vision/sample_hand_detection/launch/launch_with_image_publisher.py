# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.logging import get_logger
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    logger = get_logger('image_publisher_launch')
    
    # Declare the launch arguments for image_path and model_path
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value=os.path.join(get_package_share_directory('sample_hand_detection'), 'input_image.jpg'),
        description='Path to the image file'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/opt/model/',
        description='Path to the model file'
    )
    
    # Use LaunchConfiguration to get the values of the arguments
    image_path = LaunchConfiguration('image_path')
    model_path = LaunchConfiguration('model_path')
    
    return LaunchDescription([
        image_path_arg,
        model_path_arg,
        
        # Node for sample_hand_detection
        Node(
            package='sample_hand_detection',  # Replace with the actual package name
            executable='qrb_ros_hand_detector',  # Replace with the actual executable name
            output='screen',  # Output logs to terminal
            parameters=[
                {'model_path': model_path},  # Use the model_path argument
            ],
        ),

        # Node for image_publisher
        Node(
            package='image_publisher',  # Package name
            executable='image_publisher_node',  # Executable name
            name='image_publisher_node',  # Node name
            output='screen',  # Output logs to terminal
            parameters=[
                {'filename': image_path},  # Use the image_path argument
                {'rate': 10.0},  # Set the publishing rate to 10 Hz
            ],
        ),
    ])
