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
ROS_DISTRO="jazzy"

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

echo "==== [1/6] Install ROS and system dependencies ===="
sudo apt update
sudo apt install -y \
  ros-jazzy-rclpy \
  ros-jazzy-sensor-msgs \
  ros-jazzy-geometry-msgs \
  ros-jazzy-std-msgs \
  ros-jazzy-message-filters \
  python3-pip \
  python3-colcon-common-extensions

echo "==== [2/6] Install Orbbec camera SDK ===="
# -- Refer to [`Orbbec camera` official repository](https://github.com/orbbec/OrbbecSDK_v2)
# -- 2a. System dependencies required by OrbbecSDK ROS2 --
sudo apt install -y \
  libgflags-dev \
  nlohmann-json3-dev \
  ros-jazzy-image-transport \
  ros-jazzy-image-transport-plugins \
  ros-jazzy-compressed-image-transport \
  ros-jazzy-image-publisher \
  ros-jazzy-camera-info-manager \
  ros-jazzy-diagnostic-updater \
  ros-jazzy-diagnostic-msgs \
  ros-jazzy-statistics-msgs \
  ros-jazzy-xacro \
  ros-jazzy-backward-ros \
  libdw-dev \
  libssl-dev \
  mesa-utils \
  libgl1 \
  libgoogle-glog-dev

# -- 2b. Build OrbbecSDK ROS2 from source (v2-main branch) --
sudo apt install -y ros-jazzy-orbbec-camera ros-jazzy-orbbec-description

# -- 2c. To allow the Orbbec cameras to be recognized correctly on Linux, install the udev rules.
sudo cp /opt/ros/$ROS_DISTRO/share/orbbec_camera/udev/99-obsensor-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger


echo "==== [3/6] Install Python dependencies ===="
cd "${SAMPLE_ROOT}"
pip install -r requirements.txt

echo "==== [4/6] Prepare RM_API2 and copy Robotic_Arm ===="
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
  echo "==== [5/6] Install PyTorch CPU packages for ONNX export ===="
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

  echo "==== [5/6] Prepare DenseFusion source for export script ===="
  if [[ -z "${DENSEFUSION_PATH}" ]]; then
    DENSEFUSION_PATH="${SAMPLE_ROOT}/onnx_export/DenseFusion"
    if [[ ! -d "${DENSEFUSION_PATH}" ]]; then
      git clone https://github.com/j96w/DenseFusion.git "${DENSEFUSION_PATH}"
    fi
  fi

  # Define output paths using absolute paths to avoid cd side-effects
  YCB_MODELS_DIR="${SAMPLE_ROOT}/grasp_perception/models/ycb_models"
  ONNX_EXPORT_DIR="${SAMPLE_ROOT}/onnx_export"

  if [[ -n "${POSE_MODEL_PATH}" && -n "${REFINE_MODEL_PATH}" ]]; then
    echo "==== [6/6] Export ONNX models ===="

    # Ensure output directory exists before writing into it
    mkdir -p "${YCB_MODELS_DIR}"

    # Copy export script into DenseFusion only if not already present
    if [[ ! -f "${DENSEFUSION_PATH}/export_onnx.py" ]]; then
      cp "${ONNX_EXPORT_DIR}/export_onnx.py" "${DENSEFUSION_PATH}/"
    fi

    # Install onnx export dependency
    pip install onnxscript
    python "${DENSEFUSION_PATH}/export_onnx.py" \
      --model          "${POSE_MODEL_PATH}" \
      --refine_model   "${REFINE_MODEL_PATH}" \
      --num_points     1000 \
      --output_pose_onnx   "${YCB_MODELS_DIR}/densefusion_ycb_posenet.onnx" \
      --output_refine_onnx "${YCB_MODELS_DIR}/densefusion_ycb_refiner.onnx"

    echo "==== [6/6] ONNX export complete ===="
  else
    echo "[6/6] Skip ONNX export: POSE_MODEL_PATH and REFINE_MODEL_PATH are not both set."
    echo "      Re-run with: POSE_MODEL_PATH=<path> REFINE_MODEL_PATH=<path> bash scripts/install.bash --onnx_export"
  fi
else
  echo "[5/6] Skip ONNX steps (not enabled). Pass --onnx_export to execute steps 5-6."
fi

echo "==== [6/6] Download YOLO model and verify Orbbec camera ===="
YOLO_MODEL_DIR="${SAMPLE_ROOT}/grasp_perception/models/yolo26n_seg_models"
YOLO_MODEL_PATH="${YOLO_MODEL_DIR}/yolo26n-seg.onnx"
mkdir -p "${YOLO_MODEL_DIR}"
if [[ ! -f "${YOLO_MODEL_PATH}" ]]; then
  wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.onnx \
    -O "${YOLO_MODEL_PATH}"
else
  echo "YOLO model already exists, skipping download: ${YOLO_MODEL_PATH}"
fi

echo "Verify Orbbec camera after driver installation:"
echo "  ros2 run orbbec_camera list_devices_node"
