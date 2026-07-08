<div>
  <h1>AI Sample MiDaS + YOLO Seg Parallel</h1>
</div>

---

## 👋 Overview

The `sample_midas_yolo_parallel` sample runs MiDaS depth estimation and YOLO segmentation in parallel, then fuses both results by frame timestamp.

| Node Name | Function |
| --------- | -------- |
| image publisher / qrb ros camera / usb_cam | Provides input images from local file stream, on-board camera, or USB webcam. |
| sample midas yolo parallel | Sends each frame to two inference branches and fuses output tensors. |
| QrbRosSharedInferenceNode | Loads a single combined QNN context binary and runs both MiDaS and YOLO graphs concurrently within the shared context. Exposes one pub/sub pair per graph. |

The sample publishes:
- `/midas_depth_map` (`sensor_msgs/msg/Image`, `bgr8`)
- `/midas_depth_gray` (`sensor_msgs/msg/Image`, `mono8`)
- `/midas_yolo_overlay` (`sensor_msgs/msg/Image`, `bgr8`)

## 🔎 Table of contents

- [⚓ Used ROS Topics](#-used-ros-topics)
- [🎯 Supported targets](#-supported-targets)
- [✨ Installation](#-installation)
- [🚀 Usage](#-usage)
- [👨‍💻 Build from source](#-build-from-source)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

## ⚓ Used ROS Topics

| ROS Topic | Type | Description |
| --------- | ---- | ----------- |
| `/image_raw` | `<sensor_msgs.msg.Image>` | Input image topic for `launch_with_image_publisher.py` and `launch_with_usb_cam.py`. |
| `/cam0_stream1` | `<sensor_msgs.msg.Image>` | Input image topic for `launch_with_qrb_ros_camera.py`. |
| `/midas_inference_input_tensor` | `<qrb_ros_tensor_list_msgs.msg.TensorList>` | MiDaS inference input tensor. |
| `/yolo_seg_inference_input_tensor` | `<qrb_ros_tensor_list_msgs.msg.TensorList>` | YOLO segmentation inference input tensor. |
| `/midas_depth_map` | `<sensor_msgs.msg.Image>` | Colorized depth visualization. |
| `/midas_depth_gray` | `<sensor_msgs.msg.Image>` | Grayscale depth output. |
| `/midas_yolo_overlay` | `<sensor_msgs.msg.Image>` | Segmentation overlay with fused output. |

## 🎯 Supported targets

- Qualcomm Dragonwing IQ-9075 EVK
- Qualcomm Dragonwing IQ-8275 EVK

A combined multi-graph QNN context binary (`midas_yolo_combined.bin`) containing both the MiDaS
and YOLO segmentation graphs is required. Produce it offline using the QNN SDK's
`qnn-context-binary-utility` to merge the individual per-model context binaries.

## ✨ Installation

> [!IMPORTANT]
> The following steps need to be run on **Qualcomm Ubuntu** with **ROS 2 Jazzy**.

- Add Qualcomm PPAs and install runtime dependencies:

```bash
sudo add-apt-repository ppa:ubuntu-qcom-iot/qcom-ppa
sudo add-apt-repository ppa:ubuntu-qcom-iot/qirp
sudo apt update
sudo apt install -y \
  ros-jazzy-qrb-ros-camera \
  ros-jazzy-qrb-ros-nn-inference \
  ros-jazzy-qrb-ros-tensor-list-msgs \
  ros-jazzy-image-publisher \
  ros-jazzy-cv-bridge \
  ros-jazzy-usb-cam \
  python3-numpy
```

## 🚀 Usage

<details>
  <summary>Model preparation</summary>

Create the default model directory and place the combined context binary there:

```bash
sudo mkdir -p /opt/model
```

Place the combined multi-graph context binary (containing both MiDaS and YOLO graphs) at:

```text
/opt/model/midas_yolo_combined.bin
```

</details>

<details>
  <summary>Run with image publisher</summary>

```bash
source /opt/ros/jazzy/setup.bash
source install/local_setup.bash
ros2 launch sample_midas_yolo_parallel launch_with_image_publisher.py
```

Optional model and graph name arguments:

```bash
ros2 launch sample_midas_yolo_parallel launch_with_image_publisher.py \
  combined_model_path:=/opt/model/midas_yolo_combined.bin \
  midas_graph_name:=midas \
  yolo_graph_name:=yolov11_seg \
  image_path:=/path/to/input.jpg
```

</details>

<details>
  <summary>Run with QRB ROS camera</summary>

```bash
source /opt/ros/jazzy/setup.bash
source install/local_setup.bash
ros2 launch sample_midas_yolo_parallel launch_with_qrb_ros_camera.py
```

</details>

<details>
  <summary>Run with USB camera</summary>

```bash
source /opt/ros/jazzy/setup.bash
source install/local_setup.bash
ros2 launch sample_midas_yolo_parallel launch_with_usb_cam.py
```

> **Note:** For `usb_cam`, image quality parameters (`brightness`, `contrast`, `saturation`, `sharpness`, `gain`, and `focus`) default to `-1` (camera driver defaults). Override them as launch arguments to tune for your USB camera, for example:
>
> ```bash
> ros2 launch sample_midas_yolo_parallel launch_with_usb_cam.py brightness:=128 contrast:=64 saturation:=80 sharpness:=50 gain:=0 focus:=0
> ```

</details>

Notes:
- Both graphs must be compiled into a single combined context binary (`midas_yolo_combined.bin`).
- The default graph names are `midas` (MiDaS) and `yolov11_seg` (YOLO). Override with `midas_graph_name` and `yolo_graph_name` if your binary uses different names.
- YOLO input defaults to `float32` (`yolo_tensor_data_type:=2`, `yolo_pack_uint16_input:=false`). If your model binary expects a `uint16` input, override with `yolo_tensor_data_type:=4 yolo_pack_uint16_input:=true`.
- If you are validating with a patched `qrb_ros_nn_inference`, source that workspace before `install/local_setup.bash`.

## 👨‍💻 Build from source

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select sample_midas_yolo_parallel --executor sequential
source install/local_setup.bash
```

## 🤝 Contributing

We love community contributions. Please read [CONTRIBUTING.md](../../CONTRIBUTING.md).

## 📜 License

Project is licensed under the [BSD-3-Clause-Clear](https://spdx.org/licenses/BSD-3-Clause-Clear.html) License. See [LICENSE](../../LICENSE) for details.
