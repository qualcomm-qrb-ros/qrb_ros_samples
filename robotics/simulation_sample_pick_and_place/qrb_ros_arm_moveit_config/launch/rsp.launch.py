# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_rsp_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("rml_63", package_name="qrb_ros_arm_moveit_config").to_moveit_configs()
    ld = generate_rsp_launch(moveit_config)
    # 追加 use_sim_time 参数
    for action in ld.entities:
        if hasattr(action, 'parameters'):
            action.parameters.append({'use_sim_time': True})
    return ld
