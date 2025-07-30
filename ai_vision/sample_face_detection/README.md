<!--AI Samples <sample_face_detection>-->

# AI Samples Mediapipe Face Detection

## Overview

The ` sample_face_detection` is a Python-based face recognition ROS node that uses QNN for model inference.
The models are sourced from Qualcomm AI-hub.Can detect faces and locate facial features from image
It captures the `face_image.jpg` as input and publishes the result with the `/mediaface_det_image` and `output_image.jpg` .

For more information, please refer to [qrb_ros_samples/ai_vision/sample_face_detection at jazzy · QUIC-QRB-ROS/qrb_ros_samples](https://github.qualcomm.com/QUIC-QRB-ROS/qrb_ros_samples/tree/jazzy/ai_vision/sample_face_detection) .

Face detection camera output:
![](./resource/face_detection_cam.gif)

## Pipeline Flow For Face Detection

![](./resource/sample_face_detection_pipeline.png)

## Supported Platforms

| Hardware               | Software                                 |
| ---------------------- | ---------------------------------------- |
| IQ-9075 Evaluation Kit | Qualcomm Linux, Qualcomm Ubuntu |

## ROS Nodes Used in Face Detection Pipeline

| ROS Node         | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `qrb_ros_face_detector` | `qrb_ros_face_detector is a python-based ros jazzy packages realize classify images. This ROS node subscribes image topic, and publishs classify result topic after pre-post process. ` |
| `qrb_ros_nn_inference` | qrb_ros_nn_inference is a ROS2 package for performing neural network model, providing AI-based perception for robotics applications. source link: [qualcomm-qrb-ros/qrb_ros_nn_inference](https://github.com/qualcomm-qrb-ros/qrb_ros_nn_inference) |
| `qrb_ros_camera` | qrb_ros_camera is a ROS package to publish the images from Qualcomm CSI and GMSL cameras. source link: [qualcomm-qrb-ros/qrb_ros_camera](https://github.com/qualcomm-qrb-ros/qrb_ros_camera) |
| `image_publisher_node` | image_publisher is  a ros jazzy packages, can publish image ros topic with local path. source link: [ros-perception/image_pipeline: An image processing pipeline for ROS.](https://github.com/ros-perception/image_pipeline) |

## ROS Topics Used in Face Detection Pipeline

| ROS Topic | Type                         | Published By     |
| --------- | ---------------------------- | ---------------- |
| `/mediaface_det_image`  | `< sensor_msgs.msg.Image > ` | `qrb_ros_face_detector` |
| `image_raw`                   | `<sensor_msgs.msg.Image> `  | `image_publisher_node, camera_node` |
| `face_detector_input_tensor ` | `<qrb_ros_tensor_list_msgs.msg.TensorList> ` | `qrb_ros_face_detector`     |
| `face_detector_output_tensor ` | `<qrb_ros_tensor_list_msgs.msg.TensorList> ` | `qrb_ros_nn_inference`     |
| `face_landmark_input_tensor ` | `<qrb_ros_tensor_list_msgs.msg.TensorList> ` | `qrb_ros_face_detector`     |
| `face_landmark_output_tensor ` | `<qrb_ros_tensor_list_msgs.msg.TensorList> ` | `qrb_ros_nn_inference`     |

## Use Cases On QCLINUX

### Prerequisites

- Please refer to [Settings](https://docs.qualcomm.com/bundle/publicresource/topics/80-70018-265/download-the-prebuilt-robotics-image_3_1.html?vproduct=1601111740013072&version=1.4&facet=Qualcomm%20Intelligent%20Robotics%20Product%20(QIRP)%20SDK) to complete the device and host setup.

### On Host

**Step 1: Build sample project**

On the host machine, move to the artifacts directory and decompress the package using the `tar` command.
```bash
#generate docker image
tar -zxf qirp-sdk_<qirp_version>.tar.gz
cd <qirp_decompressed_path>/qirp-sdk
source setup.sh

#build Samples
cd <qirp_decompressed_path>/qirp-sdk/qirp_samples/ai_vision/sample_face_detection
colcon build
```

**Step 2: Package and push sample to device**

```bash
# package and push face detection models
cd <qirp_decompressed_path>/qirp-samples/demos/qrb_ros_samples/ai_vision/sample_face_detection
tar -czvf model_face_detection.tar.gz model resource/face_image.jpg
scp model_face_detection.tar.gz root@[ip-addr]:/opt/

# package and push build result of sample
cd <qirp_decompressed_path>/qirp-samples/demos/qrb_ros_samples/ai_vision/sample_face_detection/install/sample_face_detection
tar -czvf sample_face_detection.tar.gz lib share
scp sample_face_detection.tar.gz root@[ip-addr]:/opt/
```

### On Device

To Login to the device, please use the command `ssh root@[ip-addr]`

**Step 1: Install sample package and model package**

```bash
# Remount the /usr directory with read-write permissions
(ssh) mount -o remount rw /usr

# Install sample package and model package
(ssh) tar --no-same-owner -zxf /opt/sample_face_detection.tar.gz -C /usr/
(ssh) tar --no-same-owner -zxf /opt/model_face_detection.tar.gz -C /opt/
```

**Step 2: Setup runtime environment**

```bash
# Set HOME variable
(ssh) export HOME=/opt

# set SELinux to permissive mode
(ssh) setenforce 0

# setup runtime environment
(ssh) source /usr/bin/ros_setup.sh && source /usr/share/qirp-setup.sh
```

**Step 3: Run sample**

```bash
# Launch the sample face detection node with an image publisher
(ssh) ros2 launch sample_face_detection launch_with_image_publisher.py image_path:=/opt/resource/face_image.jpg model_path:=/opt/model/

# Launch the sample face detection node with an qrb_ros_camera
(ssh) ros2 launch sample_face_detection launch_with_qrb_ros_camera.py  model_path:=/opt/model/
```
Then, you can open `rviz2` on HOST and subscribe `mediaface_det_image` to get the result.
