<div >
  <h1>Samples HR pose estimation</h1>
  <p align="center">
</div>


![](./resource/result_image.jpg)

---

## 👋 Overview

The **sample_hrnet_pose_estimation** sample provides high-precision human pose estimation capabilities.
It processes input images and publishes the following ROS 2 topics:

- **`/pose_estimation_results`**: Output images with visualized pose keypoints.
- **`/pose_estimation_points`**: Raw keypoint coordinates in a structured message format.

![](./resource/sample_hrnet_pose_pipeline.png)

| Node Name                                                    | Function                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `hrnet_pose_estimation_node` | Receive the `/image_raw` topic, perform pose detection on it, and publish the `pose_estimation_results` and `pose_estimation_points` topics. |
| `image_publisher_node` | image_publisher is a ros jazzy packages, can publish image ros topic with local path. For more detail, Please refer to [image_publisher](https://github.com/ros-perception/image_pipeline). |
|`qrb_ros_camera`| The QRB ROS Camera is a ROS package to publish the images from Qualcomm CSI and GMSL cameras. For more detail, Please refer to [QRB ROS CAMERA](https://github.com/qualcomm-qrb-ros/qrb_ros_camera).|
|`qrb_ros_nn_inference`| QRB_ROS_NN_inference is a ROS2 package for performing neural network model, providing AI-based perception for robotics applications. For more detail, Please refer to [QRB ROS NN Inference](https://github.com/qualcomm-qrb-ros/qrb_ros_nn_inference). |


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

| ROS Topic                       | Type                                       | Description                                                  |
| ------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| `/image_raw `                   | `sensor_msgs.msg.Image `                   | The sample hrnet pose estimation input image.                |
| `/qrb_inference_input_tensor `  | `qrb_ros_tensor_list_msgs/msg/TensorList ` | The preprocessed image is converted into an input msg for nn inference. |
| `/qrb_inference_output_tensor ` | `qrb_ros_tensor_list_msgs/msg/TensorList ` | Message after nn inference inference                         |
| `/pose_estimation_results`      | `sensor_msgs.msg.Image`                    | Output images with visualized pose keypoints.                |
| `/pose_estimation_points`       | `geometry_msgs.msg.PolygonStamped`         | Raw keypoint coordinates in a structured message format.     |

Note: `/pose_estimation_points` contains the coordinates of 17 key points on the original image. Developers can subscribe to this topic for secondary development. Key points include: nose, left eye, right eye, etc. The specific correspondence is as follows: 

| ID  | Body Part      | ID  | Body Part      |
|-----|----------------|-----|----------------|
| 1   | Nose           | 10  | Right Wrist    |
| 2   | Right Eye      | 11  | Left Wrist     |
| 3   | Left Eye       | 12  | Right Hip      |
| 4   | Right Ear      | 13  | Left Hip       |
| 5   | Left Ear       | 14  | Right Knee     |
| 6   | Right Shoulder | 15  | Left Knee      |
| 7   | Left Shoulder  | 16  | Right Ankle    |
| 8   | Right Elbow    | 17  | Left Ankle     |
| 9   | Left Elbow     |     |                |

## 🎯 Supported targets

<table >
  <tr>
    <th>Development Hardware</th>
     <td>Qualcomm Dragonwing™ IQ-9075 EVK</td>
  </tr>
  <tr>
    <th>Hardware Overview</th>
    <th><a href="https://www.qualcomm.com/products/internet-of-things/industrial-processors/iq9-series/iq-9075"><img src="https://s7d1.scene7.com/is/image/dmqualcommprod/dragonwing-IQ-9075-EVK?$QC_Responsive$&fmt=png-alpha" width="160"></a></th>
  </tr>
  <tr>
    <th>GMSL Camera Support</th>
    <td>LI-VENUS-OX03F10-OAX40-GM2A-118H(YUV)</td>
  </tr>
</table>


## ✨ Installation

> [!IMPORTANT]
> **PREREQUISITES**: The following steps need to be run on **Qualcomm Ubuntu** and **ROS Jazzy**.<br>
> Reference [Install Ubuntu on Qualcomm IoT Platforms](https://ubuntu.com/download/qualcomm-iot) and [Install ROS Jazzy](https://docs.ros.org/en/jazzy/index.html) to setup environment. <br>
> For Qualcomm Linux, please check out the [Qualcomm Intelligent Robotics Product SDK](https://docs.qualcomm.com/bundle/publicresource/topics/80-70018-265/introduction_1.html?vproduct=1601111740013072&version=1.4&facet=Qualcomm%20Intelligent%20Robotics%20Product%20(QIRP)%20SDK) documents.

Add Qualcomm IOT PPA for Ubuntu:

coming soon

## 🚀 Usage

coming soon

## 👨‍💻 Build from source

coming soon

## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).<br>
Feel free to create an issue for bug report, feature requests or any discussion💡.

## ❤️ Contributors

Thanks to all our contributors who have helped make this project better!

<table>
  <tr>
    <td align="center"><a href="https://github.com/Ceere"><img src="https://avatars.githubusercontent.com/u/45758726?v=4" width="100" height="100" alt="Ceere"/><br /><sub><b>Ceere</b></sub></a></td>
  </tr>
</table>


## ❔ FAQs

<details>
<summary>Can multiple people be detected?</summary><br>
No, if multiple people need to be detected, please segment the images of multiple people into individual ones and perform separate detection.
</details>



## 📜 License

Project is licensed under the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) License. See [LICENSE](./LICENSE) for the full license text.
