<div>
  <h1>AI Sample MiDaS + YOLO Seg Parallel</h1>
</div>

---

## 👋 Overview

The `sample_midas_yolo_parallel` sample runs MiDaS depth estimation and YOLO segmentation in parallel, then fuses both results by frame timestamp.

| Node Name | Function |
| --------- | -------- |
| image publisher / qrb ros camera | Provides input images from local file stream or camera topics. |
| sample midas yolo parallel | Sends each frame to two inference branches and fuses output tensors. |
| qrb ros nn interface (MiDaS) | Runs MiDaS depth inference with QNN. |
| qrb ros nn interface (YOLO Seg) | Runs YOLO segmentation inference with QNN. |

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
| `/image_raw` | `<sensor_msgs.msg.Image>` | Input image topic for `launch_with_image_publisher.py`. |
| `/cam0_stream1` | `<sensor_msgs.msg.Image>` | Input image topic for `launch_with_qrb_ros_camera.py`. |
| `/midas_inference_input_tensor` | `<qrb_ros_tensor_list_msgs.msg.TensorList>` | MiDaS inference input tensor. |
| `/yolo_seg_inference_input_tensor` | `<qrb_ros_tensor_list_msgs.msg.TensorList>` | YOLO segmentation inference input tensor. |
| `/midas_depth_map` | `<sensor_msgs.msg.Image>` | Colorized depth visualization. |
| `/midas_depth_gray` | `<sensor_msgs.msg.Image>` | Grayscale depth output. |
| `/midas_yolo_overlay` | `<sensor_msgs.msg.Image>` | Segmentation overlay with fused output. |

## 🎯 Supported targets

- Qualcomm Dragonwing IQ-9075 EVK (`/opt/model/yolo11n-seg-w8a16-qcs9075.bin`)
- Qualcomm Dragonwing IQ-8275 EVK (`/opt/model/yolo11n-seg-w8a16-qcs8275-proxy.bin`)

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
  python3-numpy
```

## 🚀 Usage

<details>
  <summary>Model preparation</summary>

Create the default model directory and place both model binaries there:

```bash
sudo mkdir -p /opt/model
```

Download YOLO segmentation model:

```bash
#for IQ-8275
python3 -m qai_hub_models.models.yolov8_det.export --target-runtime qnn_context_binary  --device "QCS8275 (Proxy)" --quantize w8a16 --output-dir /opt/model/


#for IQ-9075
python3 -m qai_hub_models.models.yolov11_seg.export --target-runtime qnn_context_binary  --device "QCS9075 (Proxy) --quantize w8a16 --output-dir /opt/model/
```

Copy MiDaS binary into the same folder:

```bash
sudo cp <path_to_midas_256.bin>/midas_256.bin /opt/model/
```

Expected defaults after preparation:

```text
/opt/model/midas_256.bin
/opt/model/yolo11n-seg.bin
```

</details>

<details>
  <summary>Run with image publisher</summary>

```bash
source /opt/ros/jazzy/setup.bash
source install/local_setup.bash
ros2 launch sample_midas_yolo_parallel launch_with_image_publisher.py
```

Optional model and image arguments:

```bash
ros2 launch sample_midas_yolo_parallel launch_with_image_publisher.py \
  midas_model_path:=/opt/model/midas_256.bin \
  yolo_model_path:=/opt/model/yolo11n-seg-w8a16-qcs9075.bin \
  image_path:=/path/to/input.jpg
```

IQ8 example:

```bash
ros2 launch sample_midas_yolo_parallel launch_with_image_publisher.py \
  midas_model_path:=/opt/model/midas_256.bin \
  yolo_model_path:=/opt/model/yolo11n-seg-w8a16-qcs8275-proxy.bin \
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

Notes:
- The launch files always use `/opt/model/midas_256.bin` for MiDaS.
- The launch files auto-select the YOLO model from `/opt/model` based on target/model availability (IQ8, IQ9, then generic fallback).
- You can still override explicitly with `yolo_model_path:=...`.
- YOLO input is configured for `uint16` (`yolo_tensor_data_type:=4`, `yolo_pack_uint16_input:=true`).
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
