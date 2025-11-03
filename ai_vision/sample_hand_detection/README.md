<div align="center">
  <h1>AI Samples - Hand Landmark Detection</h1>
  <img src="./resource/result.gif" style="zoom:80%;" />      
  <a href="https://ubuntu.com/download/qualcomm-iot" target="_blank"><img src="https://img.shields.io/badge/Qualcomm%20Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white" alt="Qualcomm Ubuntu"></a>
  <a href="https://docs.ros.org/en/jazzy/" target="_blank"><img src="https://img.shields.io/badge/ROS%20Jazzy-1c428a?style=for-the-badge&logo=ros&logoColor=white" alt="Jazzy"></a>
</div>


## 👋 Overview

The `sample_hand_detection` is a Python launch file utilizing **QNN** for model inference. It demonstrates image input, AI-based inference, and real-time visualization of **hand landmark detection** results.

The model is sourced from [**MediaPipe Hand Landmark Detector**](https://aihub.qualcomm.com/iot/models/mediapipe_hand?searchTerm=hand), which predicts bounding boxes and pose skeletons of hands in an image.

<div align="center">
  <img src="./resource/arch.png" alt="architecture">
</div>

| Node Name                                                    | Function                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [qrb ros camera](https://github.com/qualcomm-qrb-ros/qrb_ros_camera) | Qualcomm ROS 2 package that captures images with parameters and publishes them to ROS topics. |
| [qrb ros nn interface](https://github.com/qualcomm-qrb-ros/qrb_ros_nn_inference) | Loads a trained AI model, receives preprocessed images, performs inference, and publishes results. |
| `qrb_ros_hand_detector`                                      | Subscribes to image topic, performs pre/post-processing, and publishes detection results. |
| `palm_preprocessor`                                      | The palm_preprocessor module is responsible for preparing input image data for palm detection. It performs image resizing, normalization, and format conversion to match the input requirements of the palm detection model. |
| `palm_postprocessor`                                      | The palm_postprocessor module interprets the raw output from the palm detection model. It extracts bounding boxes and confidence scores, applies filtering and optional non-maximum suppression (NMS), and maps the results back to the original image coordinates. |
| `landmark_preprocessor`                                      | The landmark_preprocessor module processes cropped hand regions based on palm detection results. It resizes and normalizes the hand image, and may apply geometric transformations to align the hand orientation. This prepares the input for the landmark detection model to ensure accurate keypoint estimation. |
| `landmark_preprocessor`                                      | The landmark_postprocessor module decodes the output of the landmark detection model, extracting hand keypoints. It maps these landmarks back to the original image space. |

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

| ROS Topic                          | Type                                          | Published By            |
| --------------------------------- | --------------------------------------------- | ----------------------- |
| `/handlandmark_result`            | `<sensor_msgs.msg.Image>`                    | `qrb_ros_hand_detector` |
| `/palm_detector_input_tensor`     | `<qrb_ros_tensor_list_msgs/msg/TensorList>`  | `qrb_ros_hand_detector` |
| `/palm_detector_output_tensor`    | `<qrb_ros_tensor_list_msgs/msg/TensorList>`  | `qrb_ros_nn_inference`  |
| `/landmark_detector_input_tensor` | `<qrb_ros_tensor_list_msgs/msg/TensorList>`  | `qrb_ros_hand_detector` |
| `/landmark_detector_output_tensor`| `<qrb_ros_tensor_list_msgs/msg/TensorList>`  | `qrb_ros_nn_inference`  |
| `/image_raw`                      | `<sensor_msgs.msg.Image>`                    | `qrb_ros_camera` |

---

## 🎯 Supported targets

<table >
  <tr>
    <th>Development Hardware</th>
     <td>Qualcomm Dragonwing™ IQ-9075 EVK</td>
     <td>Qualcomm Dragonwing™ IQ-8275 EVK</td>
  </tr>
  <tr>
    <th>Hardware Overview</th>
    <th><a href="https://www.qualcomm.com/products/internet-of-things/industrial-processors/iq9-series/iq-9075"><img src="https://s7d1.scene7.com/is/image/dmqualcommprod/dragonwing-IQ-9075-EVK?$QC_Responsive$&fmt=png-alpha" width="160"></a></th>
    <th>coming soon...</th>
  </tr>
  <tr>
    <th>GMSL Camera Support</th>
    <td>LI-VENUS-OX03F10-OAX40-GM2A-118H(YUV)</td>
    <td>LI-VENUS-OX03F10-OAX40-GM2A-118H(YUV)</td>
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
sudo apt install qirp-sdk
```


## 🚀 Usage

<details>
  <summary>Details</summary>

Run the sample on device

```bash
# Create necessary directories for running the sample application.
sudo mkdir -p /opt/ros/jazzy/share/sample_hand_detection/output_MediaPipeHandDetector/
sudo chmod 777 /opt/ros/jazzy/share/sample_hand_detection/output_MediaPipeHandDetector/
sudo mkdir -p /opt/ros/jazzy/share/sample_hand_detection/output_MediaPipeHandLandmarkDetector
sudo chmod 777 /opt/ros/jazzy/share/sample_hand_detection/output_MediaPipeHandLandmarkDetector
sudo chmod 777 -R /opt/ros/jazzy/share/sample_hand_detection/

# setup runtime environment
source /usr/share/qirp-setup.sh
export ROS_DOMAIN_ID=123

# Launch the sample hand detection node with an image publisher
ros2 launch sample_hand_detection launch_with_image_publisher.py
```

Then you can use `rqt` to check the `handlandmark_result`.
</details>

## 👨‍💻 Build from source

<details>
  <summary>Details</summary>

Install dependencies
```
sudo apt install qirp-sdk
```

Download the source code and build
```bash
source /usr/share/qirp-setup.sh
git clone https://github.com/qualcomm-qrb-ros/qrb_ros_samples.git
cd qrb_ros_samples/ai_vision/sample_hand_detection
colcon build
```

Run
```bash
source install/setup.bash

# Launch the sample hand detection node with an image publisher, You can replace 'image_path' with the path to your desired image.
ros2 launch sample_hand_detection launch_with_image_publisher.py image_path:=/opt/resource/input_image.jpg model_path:=/opt/model/

# Launch the sample hand detection node with qrb_ros_camera ros node.
ros2 launch sample_hand_detection launch_with_qrb_ros_camera.py model_path:=/opt/model/
```

Then you can use `rqt` to check the `handlandmark_result`.
</details>

## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).<br>
Feel free to create an issue for bug report, feature requests or any discussion💡.

## ❤️ Contributors

Thanks to all our contributors who have helped make this project better!

<table>
  <tr>
    <td align="center"><a href="https://github.com/DapengYuan-David"><img src="https://avatars.githubusercontent.com/u/129727781?v=4" width="100" height="100" alt="DapengYuan-David"/><br /><sub><b>DapengYuan-David</b></sub></a></td>
  </tr>
</table>



## ❔ FAQs

<details>
<summary>NA</summary><br>
</details>



## 📜 License

Project is licensed under the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) License. See [LICENSE](./LICENSE) for the full license text.
