<div align="center">
  <h1>AI Samples - Hand Gesture Recognition</h1>
  <!-- TODO: demo GIF pending device capture, follows the sample_hand_detection convention:
       https://github.com/qualcomm-qrb-ros/qrb_ros_samples/blob/gif/ai_vision/sample_hand_gesture_recognition/resource/result.gif -->
  <a href="https://ubuntu.com/download/qualcomm-iot" target="_blank"><img src="https://img.shields.io/badge/Qualcomm%20Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white" alt="Qualcomm Ubuntu"></a>
  <a href="https://docs.ros.org/en/jazzy/" target="_blank"><img src="https://img.shields.io/badge/ROS%20Jazzy-1c428a?style=for-the-badge&logo=ros&logoColor=white" alt="Jazzy"></a>
</div>


## 👋 Overview

The `sample_hand_gesture_recognition` is a Python launch file utilizing **QNN** for model inference. It demonstrates image input, AI-based inference, and real-time visualization of **hand gesture recognition** results.

It implements the 3-stage [**MediaPipe Hand Gesture**](https://aihub.qualcomm.com/iot/models/mediapipe_hand) pipeline:

1. **Palm detector** — locates hands in the frame and outputs palm bounding boxes.
2. **Hand landmark detector** — regresses 21 hand keypoints and the left/right-hand score from each palm ROI.
3. **Gesture classifier** — maps the normalized landmarks to one of 8 gesture classes.

Following the architecture of [`sample_hand_detection`](../sample_hand_detection/) — *large models on the NPU, post-processing math in numpy inside the node* — the two large convolutional stages (palm + landmark) run on the **NPU** through `qrb_ros_nn_inference`, while the blaze decode / NMS / sigmoid post-processing runs in numpy inside the node. The third stage is a tiny MLP, so it runs **in-node in FP32 numpy**: the w8a8 quantization of the canned gesture classifier loses too much accuracy (almost every class except `Thumb_Up` collapses to `None`), whereas the FP32 numpy path is fully accurate at negligible cost.

The 8 recognized gestures are: `None`, `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`.

```mermaid
flowchart LR
    A["Image source<br/>(image_publisher /<br/>qrb_ros_camera)"] -->|/image_raw| B["qrb_ros_gesture_recognition<br/>(palm preprocess)"]
    B -->|/palm_detector_input_tensor| C["qrb_ros_nn_inference<br/>(palm detector, NPU)"]
    C -->|/palm_detector_output_tensor| D["qrb_ros_gesture_recognition<br/>(decode / ROI / landmark preprocess)"]
    D -->|/landmark_detector_input_tensor| E["qrb_ros_nn_inference<br/>(landmark detector, NPU)"]
    E -->|/landmark_detector_output_tensor| F["qrb_ros_gesture_recognition<br/>(FP32 gesture classify + draw)"]
    F -->|/gesture_class| G["Gesture label"]
    F -->|/gesture_result_image| H["Annotated image"]
```

| Node Name                                                    | Function                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [qrb ros camera](https://github.com/qualcomm-qrb-ros/qrb_ros_camera) | Qualcomm ROS 2 package that captures images with parameters and publishes them to ROS topics. |
| [image_publisher](https://github.com/ros-perception/image_pipeline) | A ROS 2 node that publishes a still image on a loop to an image topic. |
| [qrb ros nn interface](https://github.com/qualcomm-qrb-ros/qrb_ros_nn_inference) | Loads a trained AI model, receives preprocessed tensors, performs inference on the NPU, and publishes the output tensors. |
| `qrb_ros_gesture_recognition`                                | Subscribes to the image topic; runs pre/post-processing for the palm and landmark stages, performs the in-node FP32 gesture classification, and publishes the gesture class and the visualized result image. |

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

| ROS Topic                          | Type                                          | Published By                  |
| --------------------------------- | --------------------------------------------- | ----------------------------- |
| `/gesture_result_image`           | `<sensor_msgs/msg/Image>`                     | `qrb_ros_gesture_recognition` |
| `/gesture_class`                  | `<std_msgs/msg/String>`                       | `qrb_ros_gesture_recognition` |
| `/palm_detector_input_tensor`     | `<qrb_ros_tensor_list_msgs/msg/TensorList>`   | `qrb_ros_gesture_recognition` |
| `/palm_detector_output_tensor`    | `<qrb_ros_tensor_list_msgs/msg/TensorList>`   | `qrb_ros_nn_inference`        |
| `/landmark_detector_input_tensor` | `<qrb_ros_tensor_list_msgs/msg/TensorList>`   | `qrb_ros_gesture_recognition` |
| `/landmark_detector_output_tensor`| `<qrb_ros_tensor_list_msgs/msg/TensorList>`   | `qrb_ros_nn_inference`        |
| `/image_raw`                      | `<sensor_msgs/msg/Image>`                     | `qrb_ros_camera`, `image_publisher` |

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
    <th><a href="https://www.qualcomm.com/internet-of-things/products/iq8-series/iq-8275"><img src="https://s7d1.scene7.com/is/image/dmqualcommprod/IQ8?$QC_Responsive$&fmt=png-alpha" width="160"></a></th>
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
sudo apt install ros-jazzy-sample-hand-gesture-recognition
```

## 🚀 Usage

<details>
  <summary>Details</summary>

Run the sample on device

```bash
# setup runtime environment
source /opt/ros/jazzy/setup.bash

# Launch the sample with an image publisher. You can replace 'image_path' with the path to your own image.
ros2 launch sample_hand_gesture_recognition launch_with_image_publisher.py image_path:=<path/for/your/image.jpg> model_path:=/opt/model/

# Launch the sample with the qrb_ros_camera ros node.
ros2 launch sample_hand_gesture_recognition launch_with_qrb_ros_camera.py model_path:=/opt/model/
```

Inspect the recognized gesture:

```bash
# The gesture class is published as a std_msgs/String on /gesture_class
ros2 topic echo /gesture_class

# The annotated result image (landmarks + palm box + gesture label) is on /gesture_result_image
```

**Note**
> This sample demonstrates how to build a pipeline using our ROS nodes. Due to the large data transmission of AI model inputs and outputs within ROS, prolonged operation may lead to reduced frame rates. If this happens, please relaunch the ROS nodes to restore normal performance.

</details>

## 👨‍💻 Build from source

<details>
  <summary>Details</summary>

Install dependencies
```bash
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
  ros-jazzy-qrb-ros-camera
```

### Download AI models

The pipeline uses assets from [Qualcomm AI Hub](https://aihub.qualcomm.com/iot/models/mediapipe_hand). Stages 1 & 2 run on the NPU as **QNN context binaries** (`.bin`); stage 3 runs in-node in FP32 from two PyTorch weight files. Everything lives in `/opt/model/`.

**Stage 1 & 2 (NPU) — palm & landmark detectors.**
Qualcomm AI Hub ships these as a **w8a8 QNN DLC** archive. A QNN context binary is tied to a specific HTP backend, so build the `.bin` **on / for each target board** (IQ-9075 / IQ-8275). The release asset is hosted publicly, so **no API token is required**:

```bash
# 1. Download and extract the w8a8 QNN DLC release
cd /tmp
curl -fLO https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/mediapipe_hand_gesture/releases/v0.61.0/mediapipe_hand_gesture-qnn_dlc-w8a8.zip
python3 -m zipfile -e mediapipe_hand_gesture-qnn_dlc-w8a8.zip .
# -> mediapipe_hand_gesture-qnn_dlc-w8a8/palm_detector.dlc
# -> mediapipe_hand_gesture-qnn_dlc-w8a8/hand_landmark_detector.dlc

# 2. Set up the QAIRT environment (see the QAIRT general setup guide linked below)

# 3. Build the per-target context binaries into /opt/model
sudo mkdir -p /opt/model
qnn-context-binary-generator \
    --backend /usr/lib/libQnnHtp.so \
    --model /usr/lib/libQnnModelDlc.so \
    --dlc_path /tmp/mediapipe_hand_gesture-qnn_dlc-w8a8/palm_detector.dlc \
    --binary_file palm_detector \
    --output_dir /opt/model
qnn-context-binary-generator \
    --backend /usr/lib/libQnnHtp.so \
    --model /usr/lib/libQnnModelDlc.so \
    --dlc_path /tmp/mediapipe_hand_gesture-qnn_dlc-w8a8/hand_landmark_detector.dlc \
    --binary_file hand_landmark_detector \
    --output_dir /opt/model
```

> Set up the QAIRT environment following the [QAIRT general setup guide](https://docs.qualcomm.com/doc/80-63442-10/topic/general_setup.html).
> `qnn-context-binary-generator` writes `palm_detector.bin` / `hand_landmark_detector.bin`. The `.bin` is tied to a specific backend (HTP version) and must be regenerated for every new hardware platform.

The w8a8 models have **quantized (uint8) input/output tensors**, so the node quantizes/dequantizes float32 ⇄ uint8 using the per-tensor scale/offset in `quant_params.json` (QNN convention `real = (q + offset) * scale`). These encodings are a fixed property of the released DLC, so `quant_params.json` is shipped with this sample — just copy it to `/opt/model/`:

```bash
sudo cp $(ros2 pkg prefix sample_hand_gesture_recognition)/share/sample_hand_gesture_recognition/quant_params.json /opt/model/
```

<details>
  <summary>Regenerating <code>quant_params.json</code> yourself (optional)</summary>

The scale/offset values are the quantization encodings embedded in the context binary. You can dump them for a given `.bin` with the QAIRT utility and read the per-tensor `scaleOffset` fields:

```bash
qnn-context-binary-utility --context_binary /opt/model/palm_detector.bin --json_file /tmp/palm_detector_meta.json
qnn-context-binary-utility --context_binary /opt/model/hand_landmark_detector.bin --json_file /tmp/hand_landmark_detector_meta.json
```
</details>

**Stage 3 (in-node FP32) — gesture classifier weights** and the **palm anchors** are downloaded directly:

```bash
sudo mkdir -p /opt/model && cd /opt/model
sudo wget https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/mediapipe_hand_gesture/v1/gesture_embedder.pth
sudo wget https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/mediapipe_hand_gesture/v1/gesture_classifier.pth
sudo wget https://raw.githubusercontent.com/zmurez/MediaPipePyTorch/65f2549ba35cd61dfd29f402f6c21882a32fabb1/anchors_palm.npy
```

`/opt/model/` should then contain:

```
palm_detector.bin
hand_landmark_detector.bin
quant_params.json
gesture_embedder.pth
gesture_classifier.pth
anchors_palm.npy
```

### Download the source code and build
```bash
source /opt/ros/jazzy/setup.bash
git clone https://github.com/qualcomm-qrb-ros/qrb_ros_samples.git
cd qrb_ros_samples/ai_vision/sample_hand_gesture_recognition
colcon build
```

### Run
```bash
source install/setup.bash
ros2 launch sample_hand_gesture_recognition launch_with_qrb_ros_camera.py model_path:=/opt/model/
# Or launch with a still image
ros2 launch sample_hand_gesture_recognition launch_with_image_publisher.py image_path:=<path/for/your/image.jpg> model_path:=/opt/model/
```
</details>

## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).<br>
Feel free to create an issue for bug report, feature requests or any discussion💡.

## ❤️ Contributors

Thanks to all our contributors who have helped make this project better!

<table>
  <tr>
    <td align="center"><a href="https://github.com/yli47"><img src="https://avatars.githubusercontent.com/u/312031278?v=4" width="100" height="100" alt="yli47"/><br /><sub><b>yli47</b></sub></a></td>
  </tr>
</table>

## ❔ FAQs

<details>
<summary>The publish/inference rate is only a few Hz. Why?</summary><br>
NPU (HTP) inference is serial and the palm + landmark stages run back-to-back per frame. Publishing frames faster than the NPU can consume just causes drops (the node processes one frame at a time). This is expected; keep the source frame rate modest to match actual pipeline throughput.
</details>

<details>
<summary><code>ERROR: Model format NOT support!</code> when loading the inference node</summary><br>
<code>qrb_ros_nn_inference</code> loads a <b>precompiled QNN context binary</b> (<code>.bin</code>) only; on-the-fly <code>.dlc</code> loading is not supported. Convert each DLC to a <code>.bin</code> first (see <a href="#-build-from-source">Download AI models</a>). The <code>.bin</code> is tied to a specific HTP backend and must be regenerated for every new hardware platform.
</details>

<details>
<summary>A clenched fist is sometimes labelled <code>None</code> instead of <code>Closed_Fist</code>. Is this a bug in the port?</summary><br>
No. This is a limitation of the upstream MediaPipe hand-gesture model itself, not of this ROS port or of the w8a8 quantization. On the affected frames the reference FP32 <code>hand_landmark_detector</code> + <code>canned_gesture_classifier</code> (run offline with the official ONNX/PyTorch models) produce the same <code>None</code> &gt; <code>Closed_Fist</code> ranking. Per this repository's "mimic the reference, don't add heuristics" policy, no custom correction is applied. Clear, well-framed fists are classified correctly.
</details>

<details>
<summary>Which gestures are recognized?</summary><br>
Eight classes: <code>None</code>, <code>Closed_Fist</code>, <code>Open_Palm</code>, <code>Pointing_Up</code>, <code>Thumb_Down</code>, <code>Thumb_Up</code>, <code>Victory</code>, <code>ILoveYou</code>. The recognized class is published as a <code>std_msgs/String</code> on <code>/gesture_class</code> and drawn on <code>/gesture_result_image</code>.
</details>

## 📜 License

Project is licensed under the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) License. See [LICENSE](./LICENSE) for the full license text.
