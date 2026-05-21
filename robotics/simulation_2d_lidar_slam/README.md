<div align="center">
  <h1>Simulation 2D Lidar SLAM</h1>
  <p align="center">
    <img src="https://github.com/qualcomm-qrb-ros/qrb_ros_samples/blob/gif/robotics/simulation_2d_lidar_slam/resource/simulation-2d-lidar-slam-result.gif" alt="Qualcomm QRB ROS" title="Qualcomm QRB ROS" />
  </p>
  <p>Simulation 2D Lidar SLAM sample demonstrates how to run 2D lidar SLAM in a simulated environment, enabling simultaneous localization and mapping</p>

  <a href="https://ubuntu.com/download/qualcomm-iot" target="_blank"><img src="https://img.shields.io/badge/Qualcomm%20Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white" alt="Qualcomm Ubuntu"></a>
  <a href="https://docs.ros.org/en/jazzy/" target="_blank"><img src="https://img.shields.io/badge/ROS%20Jazzy-1c428a?style=for-the-badge&logo=ros&logoColor=white" alt="Jazzy"></a>

</div>

---

## 👋 Overview

The [Simulation 2D Lidar SLAM](https://github.com/qualcomm-qrb-ros/qrb_ros_samples/tree/main/robotics/simulation_2d_lidar_slam) sample demonstrates how to run 2D lidar SLAM on Qualcomm robotics platform in a simulated environment, enabling simultaneous localization and mapping.

<div align="center">
  <img src="./resource/simulation-2d-lidar-slam-pipeline.png" alt="pipeline">
</div>

<br>

## 🔎 Table of contents
  * [APIs](#-apis)
     * [ROS interfaces](#ros-interfaces)
  * [Supported targets](#-supported-targets)
  * [Installation](#-installation)
  * [Usage](#-usage)
     * [Prerequisites](#-prerequisites)
     * [Start cartographer on device](#-start-cartographer-on-device)
     * [Launch RVIZ for visulization on host](#-launch-rviz-for-visulization-on-host)
     * [Control the AMR to start mapping on host](#-control-the-amr-to-start-mapping-on-host)
  * [Contributing](#-contributing)
  * [Contributors](#%EF%B8%8F-contributors)
  * [License](#-license)

## ⚓ APIs

### ROS interfaces

<table>
  <tr>
    <th>Name</th>
    <th>Type</th>
    <th>Description</th>
    <th>Published by</th>
  </tr>
  <tr>
    <td>/tracked_pose</td>
    <td>geometry_msgs.msg.PoseStamped</td>
    <td>The pose of the tracked frame with respect to the map frame</td>
    <td>Cartographer ROS node</td>
  </tr>
  <tr>
    <td>/scan_matched_points2</td>
    <td>sensor_msgs.msg.PointCloud2</td>
    <td>Point cloud as it was used for the purpose of scan-to-submap matching</td>
    <td>Cartographer ROS node</td>
  </tr>
  <tr>
    <td>/tf</td>
    <td>tf2_msgs.msg.TFMessage</td>
    <td>Spatial relationships between different coordinate frames</td>
    <td>Cartographer ROS node, qrb ros simulation</td>
  </tr>
  <tr>
    <td>/submap_list</td>
    <td>cartographer_ros_msgs.msg.SubmapList</td>
    <td>List of all submaps, including the pose and latest version number of each
  submap, across all trajectories</td>
    <td>Cartographer ROS node</td>
  </tr>
  <tr>
    <td>/map</td>
    <td>nav_msgs.msg.OccupancyGrid</td>
    <td>The 2D occupancy grid map of the environment</td>
    <td>Cartographer occupancy grid node</td>
  </tr>
  <tr>
    <td>/scan</td>
    <td>sensor_msg.msg.LaserScan</td>
    <td>2D laser scan data</td>
    <td>qrb ros simulation</td>
  </tr>
  <tr>
    <td>/odom</td>
    <td>nav2_msgs.msg.Odometry</td>
    <td>Robot’s estimated position and velocity over time</td>
    <td>qrb ros simulation</td>
  </tr>
</table>

## 🎯 Supported targets

<table >
  <tr>
    <th>Development Hardware</th>
    <td>Qualcomm Dragonwing™ RB3 Gen2</td>
    <td>Qualcomm Dragonwing™ IQ-9075 EVK</td>
    <td>Qualcomm Dragonwing™ IQ-8275 EVK</td>
  </tr>
  <tr>
    <th>Hardware Overview</th>
    <th><a href="https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit"><img src="https://s7d1.scene7.com/is/image/dmqualcommprod/rb3-gen2-carousel?fmt=webp-alpha&qlt=85" width="180"/></a></th>
    <th><a href="https://www.qualcomm.com/products/internet-of-things/industrial-processors/iq9-series/iq-9075"><img src="https://s7d1.scene7.com/is/image/dmqualcommprod/dragonwing-IQ-9075-EVK?$QC_Responsive$&fmt=png-alpha" width="160"></a></th>
    <th>Coming soon...</th>
  </tr>
</table>

---

## ✨ Installation

> [!IMPORTANT]
> **PREREQUISITES**: The following steps need to be run on **Qualcomm Ubuntu** and **ROS Jazzy**.<br>
> Reference [Install Ubuntu on Qualcomm IoT Platforms](https://ubuntu.com/download/qualcomm-iot) and [Install ROS Jazzy](https://docs.ros.org/en/jazzy/index.html) to setup environment. <br>
> For Qualcomm Linux, please check out the [Qualcomm Intelligent Robotics Product SDK](https://docs.qualcomm.com/bundle/publicresource/topics/80-70018-265/introduction_1.html?vproduct=1601111740013072&version=1.4&facet=Qualcomm%20Intelligent%20Robotics%20Product%20(QIRP)%20SDK) documents.

Add Qualcomm IOT PPA for Ubuntu:

```bash
sudo add-apt-repository ppa:ubuntu-qcom-iot/qcom-ppa
sudo add-apt-repository ppa:ubuntu-qcom-iot/qirp
sudo apt update
```

Install Debian package:

```bash
sudo apt install qcom-adreno-dev
sudo apt install ros-jazzy-qcom-cartographer ros-jazzy-qrb-ros-slam-msgs ros-jazzy-qcom-cartographer-ros
```

## 🚀 Usage

### 🔹 Prerequisites

#### Simulation environment setup on host

Please refer to the `Quick Start` of [QRB ROS Simulation](https://github.com/qualcomm-qrb-ros/qrb_ros_simulation) to `setup` the simulation development environment and `build` the `QRB ROS Simulation` project on your host machine. Ensure that the device and the host are on the same local network and can communicate with each other via ROS2.

#### Launch the `QRB Robot Base Mini AMR` on host

```bash
ros2 launch qrb_ros_sim_gazebo gazebo_robot_base_mini.launch.py world_model:=ionic enable_rgb_camera:=false enable_depth_camera:=false
```

Press the `Play` button to start the simulation.

### 🔹 Start cartographer on device

```bash
# Prepare for running the 2D lidar SLAM
sudo chmod 755 /opt/ros/jazzy/share/cartographer_ros/scripts/qrb_slam_service_request.sh
sudo chmod 755 /opt/ros/jazzy/share/cartographer_ros/
sudo chmod -R 755 /opt/ros/jazzy/share/cartographer_ros/map

# Run the 2D lidar SLAM
source /opt/ros/jazzy/setup.bash
ros2 launch cartographer_ros qrb_2d_lidar_slam.launch.py use_sim_time:=true
```

The output for these commands:

```bash
[INFO] [launch]: All log files can be found below /home/ubuntu/.ros/log/2026-01-17-08-23-57-220541-ubuntu-4753
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [cartographer_node-1]: process started with pid [4756]
[INFO] [cartographer_occupancy_grid_node-2]: process started with pid [4757]
[cartographer_node-1] [INFO] [1768638237.614985282] [cartographer logger]: I20260117 08:23:57.-2147483648  4756 node_main.cpp:42] Run Ignore Cycle
[cartographer_node-1] [INFO] [1768638237.630617976] [cartographer logger]: I20260117 08:23:57.-2147483648  4756 configuration_file_resolver.cc:41] Found '/opt/ros/jazzy/share/cartographer_ros/configuration_files/qrb_amr_mapping_2d.lua' for 'qrb_amr_mapping_2d.lua'.
```

### 🔹 Launch RVIZ for visulization on host

```bash
rviz2
```

Subscribe to the `/map` and `/robot_description` topic, set the Fixed Frame of Global Options to `map`.

<div align="center">
  <img src="./resource/simulation-2d-lidar-slam-rviz.png" alt="pipeline">
</div>

### 🔹 Control the AMR to start mapping on host

```bash
# Install teleop_twist_keyboard ROS package on host
sudo apt install ros-jazzy-teleop-twist-keyboard

# Run
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

---

## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).<br>
Feel free to create an issue for bug report, feature requests or any discussion💡.

## ❤️ Contributors

Thanks to all our contributors who have helped make this project better!

<table>
  <tr>
    <td align="center"><a href="https://github.com/qti-weijshen"><img src="https://avatars.githubusercontent.com/u/191950784?s=96&v=4" width="100" height="100" alt="qti-weijshen"/><br /><sub><b>qti-weijshen</b></sub></a></td>
    <td align="center"><a href="https://github.com/fulaliu"><img src="https://avatars.githubusercontent.com/u/129727781?v=4" width="100" height="100" alt="fulaliu"/><br /><sub><b>fulaliu</b></sub></a></td>
  </tr>
</table>

## 📜 License

Project is licensed under the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) License. See [LICENSE](./LICENSE) for the full license text.