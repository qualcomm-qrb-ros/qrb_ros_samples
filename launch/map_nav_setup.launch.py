# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('simulation_remote_assistant')
    pipeline_script_path = os.path.join(pkg_share, 'scripts', 'map_nav_setup.py')
    control_period_arg = DeclareLaunchArgument(
        'control_period',
        default_value='0.2',
        description='build_map_node control loop period in seconds. '
                    'Lower it (e.g. 0.05) if the robot overshoots turns and hits walls on this machine.',
    )
    run_pipeline = ExecuteProcess(
        cmd=['python3', '-u', pipeline_script_path,
             '--control-period', LaunchConfiguration('control_period')],
        output='screen',
    )
    return LaunchDescription([control_period_arg, run_pipeline])
