# Simulation Remote Assistant

## Overview
The `simulation_remote_assistant` sample application is the ROS package, which showcasing the capabilities of the Remote Assistant robot. It integrates multiple ROS 2 components to simulate an indoor autonomous mobile robot (AMR) workflow:

- Simulation: Based on [qrb_ros_simulation](https://github.com/qualcomm-qrb-ros/qrb_ros_simulation/tree/main) to simulate the robot in an office world.
- SLAM: Uses Cartographer for building maps of the environment.
- Navigation: Uses Nav2 for autonomous path planning and movement.
- Perception: Uses a YOLOv8 model for real-time object detection.

After building the map, users can interact with the robot by inputting natural language tasks such as "Go to office to check person". The robot will autonomously navigate to the specified location and perform object/person recognition tasks accordingly.


![result](resource/detection.gif) 

## Pipeline flow

The figure shows the pipeline for `simulation_remote_assistant`:

![pipeline](resource/pipeline.png) 


## Supported Platforms 

| Hardware                   | Software                                 |
| -------------------------- | ---------------------------------------- |
| Qualcomm RB3 gen2          | LE.QCROBOTICS.1.0,Canonical Ubuntu Image |
| IQ-9075 Evaluation Kit     | LE.QCROBOTICS.1.0,Canonical Ubuntu Image |
| IQ-8 Beta   Evaluation Kit | LE.QCROBOTICS.1.0,Canonical Ubuntu Image |

## ROS Nodes Used in Simulation Remote Assistant

| Node | Description |
|------|-------------|
| `qrb_ros_simulation` | Set up the Qualcomm robotic simulation environment |
| `cartographer_node` | Processes sensor data to perform SLAM |
| `cartographer_occupancy_grid_node` | Publishes a ROS occupancy grid map |
| `nav2_bringup` | Launches the Navigation2 stack for robot navigation |
| `yolo_preprocess_node ` | Execute pre/post-process for Yolo model, refer to [qrb_ros_tensor_process](https://github.com/qualcomm-qrb-ros/qrb_ros_tensor_process)
| `yolo_detection_postprocess_node ` | Same as above
| `yolo_detection_overlay_node  ` | Same as above
| `nn_inference_node ` | Performing neural network model, providing AI-based perception for robotics applications, refer to [qrb_ros_nn_inference](https://github.com/qualcomm-qrb-ros/qrb_ros_nn_inference)

## ROS Topics Used in Simulation Remote Assistant

| Topic | Direction | Type | Message Type | Description |
|-------|-----------|------|--------------|-------------|
| `/scan` | Input | Topic | `sensor_msgs/LaserScan` | 2D lidar data for SLAM and mapping|
| `/map` | Output | Topic | `nav_msgs/OccupancyGrid` | Occupancy grid map for navigation and display |
| `/tf` | Both | Topic | `tf2_msgs/TFMessage` | Transforms between coordinate frames |
| `/odom` | Input | Topic | `nav_msgs/Odometry` | Odometry data for pose estimation |
| `/cmd_vel` | Output | Topic | `geometry_msgs/Twist` | Velocity commands for robot movement |
| `/goal_pose` | Input | Topic | `geometry_msgs/PoseStamped` | Target goal pose for navigation |
| `/camera/color/image_raw` | Input | Topic | `sensor_msgs.msg.Image` | RGB image from simulation camera|
| `/qrb_inference_input_tensor` | Input | Topic | `qrb_ros_tensor_list_msgs/msg/TensorList` | yolo_preprocess_node preprocess tensor|
| `/yolo_detect_result` | Output | Topic | `vision_msgs/msg/Detection2DArray` | nn_inference_node publish the detected result|
| `/yolo_detect_tensor_output` | Output | Topic | `qrb_ros_tensor_list_msgs/msg/TensorList` | yolo_detection_postprocess_node postprocess tensor
| `/yolo_detect_overlay` | Output | Topic | `sensor_msgs.msg.Image` | The detected result with bounding box
---

## Use Case on Ubuntu and QCLINUX
<details>
  <summary>Use Case on Ubuntu</summary>

<details>
  <summary>Case: Out of box to run sample on Ubuntu</summary>

### On Host

```bash
#Launch the simulation sample env.
ros2 launch qrb_ros_sim_gazebo gazebo_robot_base_mini.launch.py \
    world_model:=office \
    initial_x:=1.0 \
    initial_y:=6.0 \
    enable_depth_camera:=false
```
### On Device

```bash
#Dowload qirp-setup.sh
(ssh) wget https://raw.githubusercontent.com/qualcomm-qrb-ros/qrb_ros_samples/refs/heads/main/tools/qirp-setup.sh -O qirp-setup.sh

#Source qirp-setup.sh
(ssh) source qirp-setup.sh

#Launch the map_nav_setup.launch.py scripts
ros2 launch simulation_remote_assistant map_nav_setup.launch.py

#Launch the node to object detection
(ssh) ros2 launch simulation_remote_assistant yolo_detectcion.launch.py

#Run the task manager to parse the location and object
(ssh) ros2 run simulation_remote_assistant task_manager_node

#Input the task commands in terminal
(ssh) go to office to check person
```

</details>
</details>

<details>
    <summary> Use Cases on QCLinux</summary>   

### Prerequisites

- Please refer to [Settings](https://docs.qualcomm.com/bundle/publicresource/topics/80-70018-265/download-the-prebuilt-robotics-image_3_1.html?vproduct=1601111740013072&version=1.4&facet=Qualcomm%20Intelligent%20Robotics%20Product%20(QIRP)%20SDK) to complete the device and host setup.

- To Login to the device, please use the command `ssh root@[ip-addr]`

### Simulation Env Setup

- Please refer to the `Quick Start` of [QRB ROS Simulation](https://github.com/qualcomm-qrb-ros/qrb_ros_simulation) to launch `QRB Robot Base AMR` on host. Ensure that the device and the host are on the same local network and can communicate with each other via ROS communication.

<details>
<summary>Case1: Out of box to run sample on QCLINUX</summary>

#### On Host

```bash
#Launch the simulation sample env.
ros2 launch qrb_ros_sim_gazebo gazebo_robot_base_mini.launch.py \
    world_model:=office \
    initial_x:=1.0 \
    initial_y:=6.0 \
    enable_depth_camera:=false
```

#### On Device

```bash
#Source qirp sdk env
(ssh) mount -o remount rw /usr
(ssh) source /usr/share/qirp-setup.sh -m

#Launch the map_nav_setup.launch.py scripts
ros2 launch simulation_remote_assistant map_nav_setup.launch.py

#Launch the node to object detection
(ssh) ros2 launch simulation_remote_assistant yolo_detectcion.launch.py

#Run the task manager to parse the location and object
(ssh) ros2 run simulation_remote_assistant task_manager_node

#Input the task commands in terminal
(ssh) go to office to check person
```
</details>

<details>
<summary>Case2: Build and run sample on QCLINUX</summary>

**Step 1: Build sample project**

#### On Host

On the host machine, move to the artifacts directory and decompress the package using the `tar` command.

```bash
#Set up qirp sdk environment
tar -zxf qirp-sdk_<qirp_version>.tar.gz
cd <qirp_decompressed_path>/qirp-sdk
source setup.sh

#Build sample
cd <qirp_decompressed_path>/qirp-sdk/qirp-samples/robotics/simulation_remote_assistant
colcon build
```

**Step 2: Package and push sample to device**

#### On Host

```bash
#Package and push build result of sample
cd <qirp_decompressed_path>/qirp-sdk/qirp-samples/robotics/simulation_remote_assistant/install/simulation_remote_assistant
tar -czf simulation_remote_assistant.tar.gz lib share
scp simulation_remote_assistant.tar.gz root@[ip-addr]:/opt/
```

**Step 3: Install sample package**

#### On Device

```bash
#Remount the /usr directory with read-write permissions
(ssh) mount -o remount rw /usr

#Install sample package
(ssh) tar --no-same-owner -zxf /opt/simulation_remote_assistant.tar.gz -C /usr/
```
**Step 4: Setup runtime environment**

#### On Device

```bash
#Setup runtime environment
(ssh) source /usr/share/qirp-setup.sh
```

**Step 5: Run sample**

#### On Host

```bash
#Launch the simulation sample env.
ros2 launch qrb_ros_sim_gazebo gazebo_robot_base_mini.launch.py \
    world_model:=office \
    initial_x:=1.0 \
    initial_y:=6.0 \
    enable_depth_camera:=false
```

**Step 5-1(Recommend): Run sample with scripts**

#### On Device

```bash
#Launch the map_nav_setup.launch.py scripts
(ssh) ros2 launch simulation_remote_assistant map_nav_setup.launch.py

#Launch the node to object detection
(ssh) ros2 launch simulation_remote_assistant yolo_detectcion.launch.py

#Run the task manager to parse the location and object
(ssh) ros2 run simulation_remote_assistant task_manager_node

#Input the task commands in terminal
(ssh) go to office to check person
```
**Step 5-2(Optional): Run sample with single node**

#### On Device

```bash
#Run the cartographer node
(ssh) ros2 launch cartographer_ros qrb_2d_lidar_slam.launch.py use_sim_time:=true

#Run the node to build map
(ssh) ros2 run simulation_remote_assistant build_map_node

#After the map is built, need to kill the cartographer node and restart it
ros2 launch cartographer_ros qrb_2d_lidar_slam.launch.py use_sim_time:=true

#Run the node to prepare navigation
(ssh) ros2 run simulation_remote_assistant nav_preparation_node

#Launch the node to object detection
(ssh) ros2 launch simulation_remote_assistant yolo_detectcion.launch.py

#Run the task manager to parse the location and object
(ssh) ros2 run simulation_remote_assistant task_manager_node

#Input the task commands in terminal
(ssh) go to office to check person
```
</details>
</details>