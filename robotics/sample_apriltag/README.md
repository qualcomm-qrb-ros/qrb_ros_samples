<div align="center">
  <h1>Apriltag Sample</h1>
  <p align="center">
  </p>
  <p>ROS package for enabling AprilTag pipeline on Qualcomm robotics platforms</p>

  <a href="https://ubuntu.com/download/qualcomm-iot" target="_blank"><img src="https://img.shields.io/badge/Qualcomm%20Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white" alt="Qualcomm Ubuntu"></a>
  <a href="https://docs.ros.org/en/jazzy/" target="_blank"><img src="https://img.shields.io/badge/ROS%20Jazzy-1c428a?style=for-the-badge&logo=ros&logoColor=white" alt="Jazzy"></a>

</div>

---

## 👋 Overview

The `sample_apriltag` is the ROS package to provide AprilTag pipeline samples for Qualcomm robotics platforms.

<div align="center">
  <img src="./resource/architecture.png" alt="architecture">
</div>

<br>

- The pipeline accepts NV12 image from `qrb_ros_camera` and converts the color space to RGB8 format.
- Then it uses `image_proc` to rectify image, outputting `/image_rect`.
- The Apriltag node detects tags and calculates their pose, which is published via `/tf`.

## 🔎 Table of contents

  * [Used ROS Topics](#-used-ros-topics)
  * [Supported targets](#-supported-targets)
  * [Installation](#-installation)
  * [Usage](#-usage)
  * [Build from source](#-build-from-source)
  * [Contributing](#-contributing)
  * [Contributors](#%EF%B8%8F-contributors)
  * [License](#-license)

## ⚓ Used ROS Topics

| ROS Topic                       | Type                                          | Description                    |
| ------------------------------- | --------------------------------------------- | ------------------------------ |
| `/apriltag/image_nv12`   | `sensor_msgs/msg/Image` | NV12 image from QRB ROS Camera |
| `/apriltag/camera_info` | `sensor_msgs/msg/CameraInfo.msg` | camera metadata information |
| `/apriltag/image_rgb8` | `sensor_msgs/msg/Image` | RGB8 image output from color space conversion |
| `/apriltag/image_rect` | `sensor_msgs/msg/Image` | rectified RGB8 image from rectify node |
| `/apriltag/detections` | `apriltag_msgs/msg/AprilTagDetectionArray` | AprilTag detection results |
| `/tf` | `tf2_msgs/msg/TFMessage` | robot transform messages |

## 🎯 Supported targets

<table >
  <tr>
    <th>Development Hardware</th>
    <td>Qualcomm Dragonwing™ RB3 Gen2</td>
    <td>Qualcomm Dragonwing™ IQ-9075 EVK</td>
  </tr>
  <tr>
    <th>Hardware Overview</th>
    <th><a href="https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit"><img src="https://s7d1.scene7.com/is/image/dmqualcommprod/rb3-gen2-carousel?fmt=webp-alpha&qlt=85" width="180"/></a></th>
    <th><a href="https://www.qualcomm.com/products/internet-of-things/industrial-processors/iq9-series/iq-9075"><img src="https://s7d1.scene7.com/is/image/dmqualcommprod/dragonwing-IQ-9075-EVK?$QC_Responsive$&fmt=png-alpha" width="160"></a></th>
  </tr>
</table>


## ✨ Installation

> [!IMPORTANT]
> **PREREQUISITES**: The following steps need to be run on **Qualcomm Ubuntu** and **ROS Jazzy**.<br>
> Reference [Install Ubuntu on Qualcomm IoT Platforms](https://ubuntu.com/download/qualcomm-iot) and [Install ROS Jazzy](https://docs.ros.org/en/jazzy/index.html) to setup environment. <br>
> For Qualcomm Linux, please check out the [Qualcomm Intelligent Robotics Product SDK](https://docs.qualcomm.com/bundle/publicresource/topics/80-70018-265/introduction_1.html?vproduct=1601111740013072&version=1.4&facet=Qualcomm%20Intelligent%20Robotics%20Product%20(QIRP)%20SDK) documents.

Add Qualcomm IOT PPA for Ubuntu:

```bash
sudo add-apt-repository ppa:ubuntu-qcom-iot/qcom-noble-ppa
sudo add-apt-repository ppa:ubuntu-qcom-iot/qirp
sudo apt update
```

Install Debian packages:

```bash
sudo apt install qcom-adreno1 weston weston-autostart ros-jazzy-sample-apriltag
```

## 🚀 Usage

### Start the AprilTag pipeline sample

```bash
sudo systemctl stop gdm.service
export XDG_RUNTIME_DIR=/run/user/1000
export WAYLAND_DISPLAY=wayland-1
sudo systemctl start weston-autostart.service

source /opt/ros/jazzy/setup.bash
ros2 launch sample_apriltag sample_apriltag.launch.py
```
## 👨‍💻 Build from source

### Dependencies

Install dependencies

```shell
sudo apt install ros-dev-tools \
  ros-jazzy-image-proc \
  ros-jazzy-apriltag-ros \
  ros-jazzy-qrb-ros-camera \
  ros-jazzy-qrb-ros-colorspace-convert
```

Download the source code and build with colcon

```bash
source /opt/ros/jazzy/setup.bash
git clone https://github.com/qualcomm-qrb-ros/qrb_ros_samples.git
colcon build --packages-up-to sample_apriltag
```

Run and debug

```bash
export XDG_RUNTIME_DIR=/run/user/1000
export WAYLAND_DISPLAY=wayland-1

source install/setup.bash
ros2 launch sample_apriltag sample_apriltag.launch.py
```

## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).<br>
Feel free to create an issue for bug report, feature requests or any discussion💡.

## ❤️ Contributors

Thanks to all our contributors who have helped make this project better!

<table>
  <tr>
    <td align="center"><a href="https://github.com/penww"><img src="https://avatars.githubusercontent.com/u/97950764?v=4" width="100" height="100" alt="penww"/><br /><sub><b>penww</b></sub></a></td>
  </tr>
</table>

## 📜 License

Project is licensed under the [BSD-3-Clause-Clear](https://spdx.org/licenses/BSD-3-Clause-Clear.html) License. See [LICENSE](../../LICENSE) for the full license text.
