<div align="center">
  <h1>Simulation Remote Assistant</h1>
  <p align="center">
    <img src="./resource/detection.gif" alt="Detection Demo" />
  </p>
  <a href="https://ubuntu.com/download/qualcomm-iot" target="_blank"><img src="https://img.shields.io/badge/Qualcomm%20Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white" alt="Qualcomm Ubuntu"></a>
  <a href="https://docs.ros.org/en/jazzy/" target="_blank"><img src="https://img.shields.io/badge/ROS%20Jazzy-1c428a?style=for-the-badge&logo=ros&logoColor=white" alt="Jazzy"></a>
  
</div>

## 👋 Overview

The `simulation_remote_assistant` sample application is the ROS package that utilizes an AMR as a remote assistant within a virtual office environment. Users can interact with the robot by inputting natural language commands, such as "Go to the office to check the person." The robot will then autonomously navigate to the specified location and perform object detection tasks as instructed.

![pipeline](resource/pipeline.png) 

| Node Name                                                    | Function                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| qrb_ros_simulation | Set up the Qualcomm robotic simulation environment, refer [qrb_ros_simulation](https://github.com/qualcomm-qrb-ros/qrb_ros_simulation). |
| nn_inference_node | Loads a trained AI model, receives preprocessed images, performs inference, and publishes results, refer [nn_inference_node](https://github.com/qualcomm-qrb-ros/qrb_ros_nn_inference). |
| yolo_preprocess_node | Execute pre/post-process for Yolo model, refer [qrb_ros_tensor_process](https://github.com/qualcomm-qrb-ros/qrb_ros_tensor_process). |
| yolo_detection_postprocess_node | Same as above. |
| yolo_detection_overlay_node | Same as above. |
| cartographer_node           | Processes sensor data to perform SLAM. |
| cartographer_occupancy_grid_node                              | Publishes a ROS occupancy grid map. |
| nav2_bringup                              | The Navigation2 stack for robot navigation. |
| build_map_node                             | Build the map of office world. |
| nav_preparation_node                             | Relocalization and launch the navigation node. |
| task_manager_node                             | Parse the input commands and execute the action. |

## 🔎 Table of contents

  * [Used ROS Topics](#-used-ros-topics)
  * [Supported targets](#-supported-targets)
  * [Installation](#-installation)
  * [Usage](#-usage)
  * [Build from source](#-build-from-source)
  * [Contributing](#-contributing)
  * [Contributors](#%EF%B8%8F-contributors)
  * [FAQs](#-faqs)
  * [License](#-license)

## ⚓ Used ROS Topics 

| ROS Topic                       | Type                                          | Description                    |
| ------------------------------- | --------------------------------------------- | ------------------------------ |
| `/scan`                   | `<sensor_msgs/LaserScan>`                   | 2D lidar data for SLAM and mapping              |
| `/map `  | `<nav_msgs/OccupancyGrid>` | Occupancy grid map for navigation             |
| `/tf` | `<tf2_msgs/TFMessage>` | Transforms between coordinate frames |
| `/odom` | `<nav_msgs/Odometry>` | Odometry data for pose estimation |
| `/cmd_vel`| `<geometry_msgs/Twist>` | Velocity commands for robot movement |
| `/goal_pose` | `<geometry_msgs/PoseStamped>` | Target goal pose for navigation |
| `/camera/color/image_raw` | `<sensor_msgs.msg.Image>` | RGB image from simulation camera|
| `/qrb_inference_input_tensor` | `<qrb_ros_tensor_list_msgs/msg/TensorList>` | yolo_preprocess_node preprocess tensor|
| `/yolo_detect_result` | `<vision_msgs/msg/Detection2DArray>` | nn_inference_node publish the detected result|
| `/yolo_detect_tensor_output` | `<qrb_ros_tensor_list_msgs/msg/TensorLis>` | yolo_detection_postprocess_node postprocess tensor
| `/yolo_detect_overlay` | `<sensor_msgs.msg.Image>` | The detected result with bounding box


## 🎯 Supported targets

<table>
  <tr>
    <th>Development Hardware</th>
    <th>Hardware Overview</th>
  </tr>
  <tr>
    <td>Qualcomm Dragonwing™ IQ-9075 EVK</td>
    <td>
      <a href="https://www.qualcomm.com/products/internet-of-things/industrial-processors/iq9-series/iq-9075">
        <img src="https://s7d1.scene7.com/is/image/dmqualcommprod/dragonwing-IQ-9075-EVK?$QC_Responsive$&fmt=png-alpha" width="160">
      </a>
    </td>
  </tr>
  <tr>
    <td>Qualcomm Robotics RB3 Gen2 Vision Kit</td>
    <td>
      <a href="https://www.qualcomm.com/products/robotics/robotics-rb3-platform">
        <img src="https://s7d1.scene7.com/is/image/dmqualcommprod/rb3-vision-kit-1" width="160">
      </a>
    </td>
  </tr>
  <tr>
    <td>Qualcomm Dragonwing™ IQ-8300</td>
    <td>
      <span>coming soon...</span>
    </td>
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

Install Debian package:

```bash
#Install the dependent debian packages
sudo apt install ros-jazzy-qrb-ros-slam-msgs
sudo apt install ros-jazzy-qcom-cartographer
sudo apt install ros-jazzy-qcom-cartographer-ros
sudo apt install ros-jazzy-qrb-ros-nn-inference
sudo apt install ros-jazzy-qrb-ros-cv-tensor-common-process
sudo apt install ros-jazzy-qrb-ros-yolo-process
sudo apt install ros-jazzy-nav2-bringup

#Install the simulation-remote-assistant debian package
sudo apt install ros-jazzy-simulation-remote-assistant
```


## 🚀 Usage

<details>
  <summary>Usage details</summary>

### Step1: Download the yolo object detection model

Reference the [qrb_ros_tensor_process](https://github.com/qualcomm-qrb-ros/qrb_ros_tensor_process) README to build and download the yolo model

```bash
#Prepare the model and move to default model path
sudo chmod -R 777 /opt/
mkdir /opt/model/
mv coco8.yaml yolov8_det_qcs9075.bin yolov8_det_qcs6490.tflite /opt/model/
```
### Step2: Run the simulation sample env on host

Reference the [qrb_ros_simulation](https://github.com/qualcomm-qrb-ros/qrb_ros_simulation/tree/main) README to prepare the simulation environment

```bash
#Env set up
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=78

#Launch the simulation office world
ros2 launch qrb_ros_sim_gazebo gazebo_robot_base_mini.launch.py \
    world_model:=office \
    initial_x:=1.0 \
    initial_y:=6.0 \
    enable_depth_camera:=false
```
### Step3: Run the remote assistant sample on device

```bash
#Env set up
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=78

#Launch the map_nav_setup.launch.py scripts
ros2 launch simulation_remote_assistant map_nav_setup.launch.py

#Launch the yolo object detection script

#Qualcomm Dragonwing™ IQ-9075 EVK 
ros2 launch simulation_remote_assistant yolo_detectcion.launch.py model:=/opt/model/yolov8_det_qcs9075.bin backend_option:=libQnnHtp.so

#Qualcomm Robotics RB3 Gen2 Vision Kit
ros2 launch simulation_remote_assistant  yolo_detectcion.launch.py model:=/opt/model/yolov8_det_qcs6490.tflite label_file:=/opt/model/coco8.yaml

#Run the task manager to parse the location and object
ros2 run simulation_remote_assistant task_manager_node

#Input the task commands in terminal
go to office to check person
```
</details>

## 👨‍💻 Build from source

<details>
  <summary>Build from source details</summary>

- Install `ros-dev-tools` .
```bash
sudo apt install ros-dev-tools
```

- Install dependency Debian packages from qcom ppa.
```bash
sudo add-apt-repository ppa:ubuntu-qcom-iot/qcom-ppa
sudo add-apt-repository ppa:ubuntu-qcom-iot/qirp
sudo apt update
```

- Download source code from qrb-ros-sample repository.
```bash
mkdir -p ~/qrb_ros_sample_ws/src && cd ~/qrb_ros_sample_ws/src
git clone -b jazzy-rel https://github.com/qualcomm-qrb-ros/qrb_ros_samples.git
```

- Build sample from source code.
```bash
cd ~/qrb_ros_sample_ws/src/qrb_ros_samples/robotics/simulation_remote_assistant
rosdep install -i --from-path ./ --rosdistro jazzy -y
colcon build
source install/setup.bash
```

- Refer to the "Usage" section to run the demo.

</details>

## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).<br>
Feel free to create an issue for bug report, feature requests or any discussion💡.

## ❤️ Contributors

Thanks to all our contributors who have helped make this project better!

<table>
  <tr>
    <td style="text-align: center;">
      <a href="https://github.com/fxt-7">
        <img src="https://avatars.githubusercontent.com/u/72394451?s=96&v=4" width="100" height="100" alt="xionfu"/>
        <br />
        <sub><b>xionfu</b></sub>
      </a>
    </td>
  </tr>
</table>


## ❔ FAQs

<details>
<summary>Which locations and objects are supported in task manager?</summary><br>
Location: office<br>
Object: person<br>
Sure, you could add new location and object in locaitons.yaml and objects.yaml
</details>


## 📜 License

Project is licensed under the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) License. See [LICENSE](./LICENSE) for the full license text.