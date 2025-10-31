

<div >
  <h1>AI Samples Object Detection	</h1>
  <p align="center">
</div>
<img src="https://github.com/qualcomm-qrb-ros/qrb_ros_samples/blob/gif/ai_vision/sample_hrnet_pose_estimation/resource/result_image.gif" style="zoom:80%;" />

---

## 👋 Overview

The `sample_object_detection` is a Python launch file utilizing QNN for model inference. It demonstrates camera data streaming, AI-based inference, and real-time visualization of object detection results.

Ultralytics YOLOv8 is a machine learning model that predicts bounding boxes, segmentation masks and classes of objects in an image.

![](./resource/pipeline.png)

| Node Name                                                    | Function                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [qrb ros camera](https://github.com/qualcomm-qrb-ros/qrb_ros_camera) | Qualcomm ROS 2 package that captures images with parameters and publishes them to ROS topics. |
| [yolo preprocess](https://github.com/qualcomm-qrb-ros/qrb_ros_tensor_process) | Subscribes to image data, reshapes/resizes it, and republishes it to a downstream topic. |
| [qrb ros nn interface](https://github.com/qualcomm-qrb-ros/qrb_ros_nn_inference) | Loads a trained AI model, receives preprocessed images, performs inference, and publishes results. |
| [yolo postprocess](https://github.com/qualcomm-qrb-ros/qrb_ros_tensor_process) | Matches inference output with yolo label files               |
| [yolo overlay](https://github.com/qualcomm-qrb-ros/qrb_ros_tensor_process) | Subscribes yolo postprocess and image data, show the object detect results with ros topic |

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

| ROS Topic                      | Type                                          | Published By                       |
| ------------------------------ | --------------------------------------------- | ---------------------------------- |
| `/camera/color/image_raw `     | `< sensor_msgs.msg.Image> `                   | `qrb_ros_camera `                  |
| `/qrb_inference_input_tensor ` | `< qrb_ros_tensor_list_msgs/msg/TensorList> ` | `yolo_preprocess_node `            |
| `/yolo_detect_result `         | `<vision_msgs/msg/Detection2DArray> `         | `nn_inference_node `               |
| `/yolo_detect_tensor_output `  | `< qrb_ros_tensor_list_msgs/msg/TensorList> ` | `yolo_detection_postprocess_node ` |
| `/yolo_detect_overlay `        | `< sensor_msgs.msg.Image> `                   | `yolo_detection_overlay_node `     |

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
sudo apt install ros-jazzy-sample-object-detection
```

## 🚀 Usage

<details>
  <summary>Usage details</summary>

Download the yolo object object model

Reference the [qrb_ros_tensor_process](https://github.com/qualcomm-qrb-ros/qrb_ros_tensor_process) README to build and download the yolo model

```
#when download yolo model , please using qnn_context_binary and device like bellow for IQ-8275 
 
python3 -m qai_hub_models.models.yolov8_det.export --target-runtime qnn_context_binary  --device "QCS8275 (Proxy)"
```

Run the sample env on device


```bash
#Prepare above model and move to default model path
mkdir /opt/model/
mv coco8.yaml yolov8_det_qcs9075.bin /opt/model/

source /opt/ros/jazzy/setup.bash
ros2 launch sample_object_detection launch_with_qrb_ros_camera.py  model:=<the device model>
```

The output for these commands:

```
root@qcs8300-ride-sx:/root# ros2 launch sample_object_detection launch_with_qrb_ros_camera.py
[INFO] [launch]: All log files can be found below /opt/.ros/log/1970-01-01-00-57-22-354131-qcs8300-ride-sx-57950
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [component_container-1]: process started with pid [58040]
[component_container-1] [INFO] [0000003442.901772236] [yolo_node_container]: Load Library: /usr/lib/libqrb_ros_yolo_process_component.so
[component_container-1] [INFO] [0000003442.955868330] [yolo_node_container]: Found class: rclcpp_components::NodeFactoryTemplate<qrb_ros::yolo_process::YoloDetOverlayNode>
[component_container-1] [INFO] [0000003442.955984632] [yolo_node_container]: Instantiate class: rclcpp_components::NodeFactoryTemplate<qrb_ros::yolo_process::YoloDetOverlayNode>
[component_container-1] [INFO] [0000003442.978678955] [yolo_detection_overlay_node]: init done~
[INFO] [launch_ros.actions.load_composable_nodes]: Loaded node '/yolo_detection_overlay_node' in container '/yolo_node_container'
[component_container-1] [INFO] [0000003442.993816298] [yolo_node_container]: Found class: rclcpp_components::NodeFactoryTemplate<qrb_ros::yolo_process::YoloDetOverlayNode>
[component_container-1] [INFO] [0000003442.993910569] [yolo_node_container]: Found class: rclcpp_components::NodeFactoryTemplate<qrb_ros::yolo_process::YoloDetPostProcessNode>
[component_container-1] [INFO] [0000003442.993928850] [yolo_node_container]: Instantiate class: rclcpp_components::NodeFactoryTemplate<qrb_ros::yolo_process::YoloDetPostProcessNode>
[component_container-1] [INFO] [0000003443.004949215] [yolo_detection_postprocess_node]: label file path: /opt/coco8.yaml
[component_container-1] [INFO] [0000003443.005031298] [yolo_detection_postprocess_node]: iou_thres: 0.500000
[component_container-1] [INFO] [0000003443.005055725] [yolo_detection_postprocess_node]: score_thres: 0.700000
[component_container-1] YAML Exception: bad file: /opt/coco8.yaml
[component_container-1] [INFO] [0000003443.008653955] [yolo_detection_postprocess_node]: init done~
[INFO] [launch_ros.actions.load_composable_nodes]: Loaded node '/yolo_detection_postprocess_node' in container '/yolo_node_container'
[component_container-1] [INFO] [0000003443.024663017] [yolo_node_container]: Load Library: /usr/lib/libqrb_ros_inference_node.so
[component_container-1] [INFO] [0000003443.031113798] [yolo_node_container]: Found class: rclcpp_components::NodeFactoryTemplate<qrb_ros::nn_inference::QrbRosInferenceNode>
[component_container-1] [INFO] [0000003443.031221975] [yolo_node_container]: Instantiate class: rclcpp_components::NodeFactoryTemplate<qrb_ros::nn_inference::QrbRosInferenceNode>
[component_container-1] [QRB INFO] Loading model from binary file: /opt/model/yolov8_det_qcs9075.bin
[component_container-1]  <W> Initializing HtpProvider
[component_container-1] [QRB INFO] /usr/lib/libQnnHtp.so initialize successfully
[component_container-1] /prj/qct/webtech_scratch20/mlg_user_admin/qaisw_source_repo/rel/qairt-2.35.0/release/snpe_src/avante-tools/prebuilt/dsp/hexagon-sdk-5.4.0/ipc/fastrpc/rpcmem/src/rpcmem_android.c:38:dummy call to rpcmem_init, rpcmem APIs will be used from libxdsprpc
[component_container-1] [QRB INFO] Qnn device initialize successfully
[component_container-1]  <W> No usable logger handle was found
[component_container-1]  <W> Logs will be sent to the system's default channel
[component_container-1]  <W> No usable logger handle was found
[component_container-1]  <W> No usable logger handle was found
[component_container-1]  <W> Logs will be sent to the system's default channel
[component_container-1] [QRB INFO] Initialize Qnn graph from binary file successfully
[component_container-1] [INFO] [0000003443.381394996] [nn_inference_node]: Inference init successfully!
[INFO] [launch_ros.actions.load_composable_nodes]: Loaded node '/nn_inference_node' in container '/yolo_node_container'
[component_container-1] [INFO] [0000003443.399924527] [yolo_node_container]: Load Library: /usr/lib/libqrb_ros_cv_tensor_common_process_component.so
[component_container-1] [INFO] [0000003443.413261611] [yolo_node_container]: Found class: rclcpp_components::NodeFactoryTemplate<qrb_ros::cv_tensor_common_process::CvTensorCommonProcessNode>
[component_container-1] [INFO] [0000003443.413547809] [yolo_node_container]: Instantiate class: rclcpp_components::NodeFactoryTemplate<qrb_ros::cv_tensor_common_process::CvTensorCommonProcessNode>
[component_container-1] [INFO] [0000003443.466812861] [yolo_preprocess_node]: params list -> resoltuion: 640x640, tensor format: nhwc, dtype: float32, normalize: 1
[INFO] [launch_ros.actions.load_composable_nodes]: Loaded node '/yolo_preprocess_node' in container '/yolo_node_container'
```

Then you can check ROS topics with the name`/yolo_detect_overlay` in  rviz2

</details>

## 👨‍💻 Build from source

<details>
  <summary>Build from source details</summary>
Install dependencies

```
sudo apt install ros-jazzy-rclpy \
  ros-jazzy-sensor-msgs \
  ros-jazzy-std-msgs \
  ros-jazzy-cv-bridge \
  ros-jazzy-ament-index-python \
  ros-jazzy-qrb-ros-tensor-list-msgs \
  python3-opencv \
  python3-numpy \
  ros-jazzy-image-publisher \
  ros-jazzy-qrb-ros-nn-inference \
  ros-jazzy-qrb-ros-camera \
```

Download the source code and build with colcon

```bash
source /opt/ros/jazzy/setup.bash
git clone https://github.com/qualcomm-qrb-ros/qrb_ros_samples.git
cd ai_vision/sample_object_detection
colcon build
```

</details>

## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).<br>
Feel free to create an issue for bug report, feature requests or any discussion💡.

## ❤️ Contributors

Thanks to all our contributors who have helped make this project better!

<table>
  <tr>
    <td align="center"><a href="https://github.com/quic-fulan"><img src="https://avatars.githubusercontent.com/u/129727781?v=4" width="100" height="100" alt="quic-fulan"/><br /><sub><b>quic-fulan</b></sub></a></td>
  </tr>
</table>



## ❔ FAQs

<details>
<summary>NA</summary><br>
</details>



## 📜 License

Project is licensed under the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) License.