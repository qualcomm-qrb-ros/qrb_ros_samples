# Follow Me

<img src="./resources/follow_me_gif.gif" style="zoom:80%;" />

## 👋 Overview

The `Follow Me` sample is a AMR to detect, track, and follow a moving person in real time. It integrates sensor emulation and motion control to  follow human-following behavior in real environments.

![](./resources/image_1.png)

| Node Name            | Function                                                     |
| -------------------- | ------------------------------------------------------------ |
| camera node          | Publishes camera frames rgb image data to a ROS topic in Gazebo |
| depth camera node    | Publishes camera frames depth image data to a ROS topic in Gazebo |
| Root base node       | Subscribes to control commands and coordinates motion with the AMR in Gazebo. |
| Follow me Tracker    | Subscribes camera info and run model to target detection on Device |
| Follow me Controller | Publishes movement execution /cmd_vel commands on Device     |

## 🔎 Table of contents

  * [Used ROS Topics](#-used-ros-topics)
  * [Supported targets](#-supported-targets)
  * [Installation](#-installation)
  * [Usage](#-usage)
  * [Build from source](#build-from-source)
  * [Contributing](#-contributing)
  * [Contributors](#%EF%B8%8F-contributors)
  * [FAQs](#-faqs)
  * [License](#-license)

## ⚓ Used ROS Topics 

| ROS Topic                  | Type                          | Description              |
| -------------------------- | ----------------------------- | ------------------------ |
| `/camera/color/image_raw ` | `< sensor_msgs.msg.Image> `   | image rgb topic          |
| `/camera/depth/image_raw ` | `< sensor_msgs.msg.Image> `   | image depth topic        |
| `/cmd_vel `                | `< geometry_msgs/msg/Twist> ` | movement execution topic |

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
    <th>coming soon...</th>
  </tr>
</table>


## ✨ Installation

> [!IMPORTANT]
> **PREREQUISITES**: The following steps need to be run on **Qualcomm Ubuntu** and **ROS Jazzy**.<br>
> For Qualcomm Linux, please check out the [Qualcomm Intelligent Robotics Product SDK](https://docs.qualcomm.com/bundle/publicresource/topics/80-70018-265/introduction_1.html?vproduct=1601111740013072&version=1.4&facet=Qualcomm%20Intelligent%20Robotics%20Product%20(QIRP)%20SDK) documents to prepare  complete the device.

Add Qualcomm IOT PPA for Ubuntu:

```bash
sudo add-apt-repository ppa:ubuntu-qcom-iot/qcom-ppa
sudo add-apt-repository ppa:ubuntu-qcom-iot/qirp
sudo apt update
```

Download model files:
```bash
# Download label file
wget https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt
# Download model weights
wget https://github.com/dog-qiuqiu/FastestDet/blob/main/example/ncnn/FastestDet.bin
wget https://github.com/dog-qiuqiu/FastestDet/blob/main/example/ncnn/FastestDet.param
# copy model files
cp FastestDet.bin FastestDet.param coco-labels-2014_2017.txt /usr/share/follow-me/model/
```


Install Debian packages:

```bash
sudo apt install ros-jazzy-ncnn
sudo apt install ros-jazzy-follow-me
```

## 🚀 Usage

<details>
  <summary>Usage details</summary>

### On Device

To Login to the device, please use the command `ssh root@[ip-addr]`.

```
# run camera node on device.
source /usr/share/qirp-setup.sh
source /opt/ros/jazzy/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py color_width=848 color_height=480 color_fps=15

# run follow me on device.
source /usr/share/qirp-setup.sh
follow_me
```

</details>

## 👨‍💻 Build from source

### Dependencies
Install dependencies

```bash
sudo add-apt-repository ppa:ubuntu-qcom-iot/qcom-ppa
sudo add-apt-repository ppa:ubuntu-qcom-iot/qirp
sudo apt update

sudo apt install ncnn-dev
```
Download model files:
```bash
# Download label file
wget https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt
# Download model weights
wget https://github.com/dog-qiuqiu/FastestDet/blob/main/example/ncnn/FastestDet.bin
wget https://github.com/dog-qiuqiu/FastestDet/blob/main/example/ncnn/FastestDet.param
# copy model files
cp FastestDet.bin FastestDet.param coco-labels-2014_2017.txt /usr/share/follow-me/model/
```

Download the source code and build with colcon
```bash
source /opt/ros/jazzy/setup.bash
git clone https://github.com/qualcomm-qrb-ros/qrb_ros_samples.git
cd qrb_ros_samples/robotisc/sample_followme/
colcon build
```

Run

```bash
cd ./install/follow_me/bin/
.follow_me
```


## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).<br>
Feel free to create an issue for bug report, feature requests or any discussion💡.

## ❤️ Contributors

Thanks to all our contributors who have helped make this project better!

<table>
  <tr>
    <td align="center"><a href="https://github.com/quic-fulan"><img src="https://avatars.githubusercontent.com/u/129727781?v=4" width="100" height="100" alt="quic-fulan"/><br /><sub><b>quic-fulan</b></sub></a></td>
    <td align="center"><a href="https://github.com/yuji-quic"><img src="https://avatars.githubusercontent.com/u/33081913?v=4" width="100" height="100" alt="yuji-quic"/><br /><sub><b>yuji-quic</b></sub></a></td>
  </tr>
</table>

## ❔ FAQs

<details>
<summary>Why only have two pre-set path in the sample?</summary><br>
This sample is intended to demonstrate our existing "follow-me" functionality and the simulation environment. Therefore, additional scenes are not configured. If needed, you can modify the world model file (for example: warehouse_followme_path2 in qrb ros simulation project) to change the character’s movement trajectory.
</details>

## 📜 License

Project is licensed under the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) License. See [LICENSE](./LICENSE) for the full license text.



