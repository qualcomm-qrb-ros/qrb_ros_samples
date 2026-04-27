#!/usr/bin/env bash
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Optional environment variables:
#   RM_API2_PATH: existing RM_API2 repo path; if unset, clone to /tmp/RM_API2.
#   DENSEFUSION_PATH: existing DenseFusion repo path; if unset, clone to ${SAMPLE_ROOT}/onnx_export/DenseFusion.
#   POSE_MODEL_PATH: DenseFusion PoseNet .pth path (required for ONNX export step).
#   REFINE_MODEL_PATH: DenseFusion Refiner .pth path (required for ONNX export step).
RM_API2_PATH="${RM_API2_PATH:-}"
DENSEFUSION_PATH="${DENSEFUSION_PATH:-}"
POSE_MODEL_PATH="${POSE_MODEL_PATH:-}"
REFINE_MODEL_PATH="${REFINE_MODEL_PATH:-}"
ENABLE_ONNX_EXPORT=false

for arg in "$@"; do
  case "${arg}" in
    --onnx_export)
      ENABLE_ONNX_EXPORT=true
      ;;
    -h|--help)
      echo "Usage: bash scripts/install.bash [--onnx_export]"
      echo "  --onnx_export   Execute steps 4-6 (ONNX export related steps)."
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}"
      echo "Usage: bash scripts/install.bash [--onnx_export]"
      exit 1
      ;;
  esac
done

echo "[1/6] Install ROS and system dependencies"
sudo apt update
# sudo apt install -y \
#   ros-jazzy-rclpy \
#   ros-jazzy-sensor-msgs \
#   ros-jazzy-geometry-msgs \
#   ros-jazzy-std-msgs \
#   ros-jazzy-message-filters \
#   python3-pip \
#   python3-colcon-common-extensions

echo "[2/6] Install Python dependencies"
cd "${SAMPLE_ROOT}"
pip install -r requirements.txt

echo "[3/6] Prepare RM_API2 and copy Robotic_Arm"
if [[ -z "${RM_API2_PATH}" ]]; then
  RM_API2_PATH="/tmp/RM_API2"
  if [[ ! -d "${RM_API2_PATH}" ]]; then
    git clone https://github.com/RealManRobot/RM_API2.git "${RM_API2_PATH}"
  fi
fi
git -C "${RM_API2_PATH}" fetch --all --tags
git -C "${RM_API2_PATH}" checkout f57fe52f767e76ce3670c526b554120472ec5e8c
cp -r "${RM_API2_PATH}/Demo/RMDemo_Python/RMDemo_Gripper/src/Robotic_Arm" \
  "${SAMPLE_ROOT}/grasp_execution/grasp_execution/src/"

if [[ "${ENABLE_ONNX_EXPORT}" == "true" ]]; then
  echo "[4/6] Install PyTorch CPU packages for ONNX export"
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

  echo "[5/6] Prepare DenseFusion source for export script"
  if [[ -z "${DENSEFUSION_PATH}" ]]; then
    DENSEFUSION_PATH="${SAMPLE_ROOT}/onnx_export/DenseFusion"
    if [[ ! -d "${DENSEFUSION_PATH}" ]]; then
      git clone https://github.com/j96w/DenseFusion.git "${DENSEFUSION_PATH}"
    fi
  fi

  if [[ -n "${POSE_MODEL_PATH}" && -n "${REFINE_MODEL_PATH}" ]]; then
    echo "[5/6] Export ONNX models"
    cd "${SAMPLE_ROOT}/onnx_export"
    cp export_onnx.py DenseFusion
    python export_onnx.py \
      --model "${POSE_MODEL_PATH}" \
      --refine_model "${REFINE_MODEL_PATH}" \
      --num_points 1000 \
      --output_pose_onnx ../grasp_perception/models/ycb_models/densefusion_ycb_posenet.onnx \
      --output_refine_onnx ../grasp_perception/models/ycb_models/densefusion_ycb_refiner.onnx
  else
    echo "[5/6] Skip ONNX export: set POSE_MODEL_PATH and REFINE_MODEL_PATH to enable."
  fi

  echo "[6/6] Reminder: download YOLO and verify Orbbec camera"
  echo "Download YOLO ONNX from Qualcomm AI Hub:"
  echo "  https://aihub.qualcomm.com/iot/models/yolov11_seg?searchTerm=yolov"
  echo "Place it at:"
  echo "  ${SAMPLE_ROOT}/grasp_perception/models/yolo26n_seg_models/yolo26n-seg.onnx"
  echo
  echo "Verify Orbbec camera after driver installation:"
  echo "  ros2 run orbbec_camera list_devices_node"
else
  echo "[4/6] Skip ONNX steps (not enabled). Pass --onnx_export to execute steps 4-6."
fi
