#!/usr/bin/env bash
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /opt/ros/jazzy/setup.bash
# source ~/miniconda3/bin/activate onepose
source "${SAMPLE_ROOT}/install/setup.bash"

ros2 launch orbbec_camera gemini_330_series.launch.py \ 
    depth_registration:=true align_mode:=SW \ 
    align_target_stream:=COLOR \
    color_width:=640 color_height:=480 color_fps:=15 \
    depth_width:=640 \
    depth_height:=480 \
    depth_fps:=15
sleep 3

python grasp_execution/grasp_execution/src/main.py --ip 192.168.1.18 --pose-topic /pose_estimation_result
sleep 3

ros2 launch grasp_perception inference_with_camera_stream.launch.py