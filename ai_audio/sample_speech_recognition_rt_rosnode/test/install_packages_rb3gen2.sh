# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear


# Copyright 2015 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# set -eo pipefail
export DEBIAN_FRONTEND=noninteractive

# system baseic and Locale
echo "[1] Updating package list..."
sudo apt-get update -y
sudo apt-get install -y locales software-properties-common curl build-essential python3-pip

# verify in env
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: Must into the venv then run this script!"
    echo "please run: source ~/venv_ros/bin/activate"
    exit 1
fi
echo "[2] Detected venv at: $VIRTUAL_ENV"
echo "All Python Lib install into venv."

# set env
# echo "[3] Setting up locale..."
# sudo locale-gen en_US en_US.UTF-8
# sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
# export LANG=en_US.UTF-8

# add ROS 2 source & install ROS Jazzy
# echo "[4] Installing ROS 2 Jazzy..."
# sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
#   -o /usr/share/keyrings/ros-archive-keyring.gpg

# echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
# http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
# | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null

# sudo apt-get update -y
# sudo apt-get install -y ros-jazzy-ros-base

# install HDF5
echo "[5] Installing PortAudio & HDF5 system libs..."
sudo apt-get install -y libhdf5-dev libportaudio2 libportaudiocpp0 portaudio19-dev

# install colcon in venv
echo "[6] Installing colcon-common-extensions into venv..."
python3 -m pip install -U pip wheel setuptools
python3 -m pip install -U colcon-common-extensions

# install Python dependence
echo "[7] Installing Python dependencies into venv..."

python3 -m pip install \
  "h5py==3.14.0" \
  "scipy==1.15.3" \
  "sounddevice==0.5.2" \
  "samplerate==0.1.0" \
  "openai-whisper==20240930" \
  "librosa" \
  "numba==0.62.0" \
  "coverage==7.10.7"

python3 -m pip install "torch==2.5.1"
python3 -m pip install "tensorflow==2.19.0"
python3 -m pip install "qai_hub_models==0.34.1"

echo "All packages installed successfully!"

# verify venv
echo "[8] Validating sounddevice import..."
python3 - <<'PY'
import sys, sounddevice as sd
print("[OK] Python   ->", sys.executable)
print("[OK] sd ver.  ->", sd.__version__)
PY
