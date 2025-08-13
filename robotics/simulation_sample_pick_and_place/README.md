

<div >
  <h1>Simulation Sample Pick and Place</h1>
  <p align="center">
</div>

![](./resource/pick_and_place.gif)

---

## 👋 Overview

- The RML-63 Robotic Arm Pick and Place Demo is a C++-based robotic manipulation ROS2 node that demonstrates autonomous pick-and-place operations using MoveIt2 for motion planning and Gazebo for physics simulation.

![image-20250723181610392](./resource/pick_and_place_architecture.jpg)

| Node Name                                                    | Function                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| qrb_ros_simulation | Set up the Qualcomm robotic simulation environment, refer [qrb_ros_simulation](https://github.com/qualcomm-qrb-ros/qrb_ros_simulation). |
| qrb_ros_arm_pick_and_place     | Predefined pick and place positions. ROS2 launch.py support for configuration parameters. |


## 🔎 Table of contents

- [👋 Overview](#-overview)
- [🔎 Table of contents](#-table-of-contents)
- [⚓ Used ROS Topics](#-used-ros-topics)
- [🎯 Supported targets](#-supported-targets)
- [✨ Installation](#-installation)
- [🚀 Usage](#-usage)
- [👨‍💻 Build from source](#-build-from-source)
- [🤝 Contributing](#-contributing)
- [❤️ Contributors](#️-contributors)
- [❔ FAQs](#-faqs)
- [📜 License](#-license)

## ⚓ Used ROS Topics 

| ROS Topic                       | Type                                          | Description                    |
| ------------------------------- | --------------------------------------------- | ------------------------------ |
| `/joint_states`                   | `<sensor_msgs/msg/JointState> `                   | 	Real-time joint position, velocity, and effort data for all robot jointscontrolinfo              |
| `/hand_controller/controller_state` | `<control_msgs.msg.ControllerState>` |	Current state and status information of the gripper controller |
| `/hand_controller/joint_trajectory` |	`<trajectory_msgs.msg.JointTrajectory>` |	Trajectory commands sent to gripper joints for motion execution |
| `/rm_group_controller/controller_state` |	`<control_msgs.msg.ControllerState>` |	Current state and status information of the robotic arm controller |
| `/rm_group_controller/joint_trajectory` |	`<trajectory_msgs.msg.JointTrajectory>` |	Trajectory commands sent to arm joints for motion execution |
| `/robot_description` |	`<std_msgs.msg.String>` |	URDF robot description in XML format for robot modeling and visualization |
| `/robot_description_semantic` |	`<std_msgs.msg.String>` |	SRDF semantic robot description for MoveIt planning and configuration |

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
    <td>Qualcomm Dragonwing™ RB3 Gen2</td>
    <td>
      <a href="https://www.qualcomm.com/products/internet-of-things/industrial-processors/rb3-series/rb3-gen2">
        <img src="https://s7d1.scene7.com/is/image/dmqualcommprod/rb3-vision-kit-1" width="160">
      </a>
    </td>
  </tr>
    <tr>
    <td>Qualcomm Dragonwing™ IQ-8275</td>
    <td>
      <a href="https://www.qualcomm.com/products/internet-of-things/industrial-processors/iq8-series">
        <img src="data:image/png;base64,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" width="160">
      </a>
    </td>
  </tr>
</table>

## ✨ Installation

> [!IMPORTANT]

> **PREREQUISITES**: 
For Qualcomm Linux, please check out the [Qualcomm Intelligent Robotics Product SDK](https://docs.qualcomm.com/bundle/publicresource/topics/80-70020-265/quick_start.html?vproduct=1601111740013072&version=1.5&facet=Qualcomm%20Intelligent%20Robotics%20SDK#setup-demo-qs) documents.



## 🚀 Usage

<details>
  <summary>Usage details</summary>

- Launch Gazebo and Rviz on HOST docker

Please refer to the Quick Start of [QRB ROS Simulation](https://github.com/qualcomm-qrb-ros/qrb_ros_simulation) to launch `QRB Robot ARM` on host. Use the same local network and same `ROS_DOMAIN_ID` to ensure that the device can communicate with each other via ROS communication.

You can also launch Gazebo with the following command:
```bash
ros2 launch qrb_ros_sim_gazebo gazebo_rml_63_gripper.launch.py
```
Click play button in Gazebo after rendered the world environment, and then use the following command to launch controller.
```bash
ros2 launch qrb_ros_sim_gazebo gazebo_rml_63_gripper_load_controller.launch.py
``` 
Make sure that after you started Gazebo and Rviz in the host Docker, you can select arm predefined state `ready` and `home` in Rviz to start the arm motion.

- On device

Then you can launche the MoveIt! configuration and launch the demo launch file to start the arm motion.
```bash
source /usr/share/qirp-setup.sh
ros2 launch simulation_sample_pick_and_place simulation_sample_pick_and_place.launch.py
```
If arm motion work normall， open another terminal, , you can use the following command to start the pick and place node.
```bash
source /usr/share/qirp-setup.sh
ros2 run simulation_sample_pick_and_place qrb_ros_arm_pick_place
```

Then you can view the arm execute pick and place operation in Gazebo and Rviz.

</details>

## 👨‍💻 Build from source

Coming soon ...

## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).<br>
Feel free to create an issue for bug report, feature requests or any discussion💡.

## ❤️ Contributors

Thanks to all our contributors who have helped make this project better!

<table>
  <tr>
    <td style="text-align: center;">
      <a href="https://github.com/DotaIsMind">
        <img src="https://github.com/DotaIsMind.png" width="100" height="100" alt="teng"/>
        <br />
        <sub><b>teng</b></sub>
      </a>
    </td>
  </tr>
</table>


## ❔ FAQs


## 📜 License

Project is licensed under the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) License. See [LICENSE](../../LICENSE) for the full license text.