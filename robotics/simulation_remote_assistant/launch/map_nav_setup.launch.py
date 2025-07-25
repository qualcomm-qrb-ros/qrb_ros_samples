# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
# SPDX-License-Identifier: BSD-3-Clause-Clear

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('simulation_remote_assistant')
    pipeline_script_path = os.path.join(pkg_share, 'scripts', 'map_nav_setup.py')
    run_pipeline = ExecuteProcess(
        cmd=['python3', pipeline_script_path],
        output='screen',
    )
    return LaunchDescription([run_pipeline])