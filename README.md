# QRB ROS Samples

## Overview

This repository is a comprehensive collection of QRB ROS (Robot Operating System) example codes. It serves as a valuable resource for developers and enthusiasts looking to explore and implement QRB functionalities within the ROS framework. Each example is designed to demonstrate specific features and use cases, providing a practical guide to enhance your understanding and application of QRB in ROS environments.



The `main` branch serves as the development branch and includes all samples currently under active development. It is primarily supported on Ubuntu by default.    For stable releases, please refer to the `jazzy-rel` branch.

## List of AI Samples

| Sample                                                       | Peripherals required | Develop Ready for Ubuntu | RB3 Gen2 Vision Kit | IQ-9075 Evaluation Kit | IQ-8 Beta   Evaluation Kit | Description                                                  |
| ------------------------------------------------------------ | -------------------- | ------------------- | ---------------------- | -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Face detection](ai_vision/sample_face_detection/)           | QRB ROS Camera          | Y                  | N                   | Y                      | N                          | The Face detection is a machine learning pipeline that predicts bounding boxes and key point of face in an image. For model information, please refer to [MediaPipe-Face-Detection](https://huggingface.co/qualcomm/MediaPipe-Face-Detection). |
| [Hand detection](ai_vision/sample_hand_detection/)           | QRB ROS Camera      | Y                  | N                   | Y                      | N                          | The Hand detection is a machine learning pipeline that predicts bounding boxes and pose skeletons of hands in an image. For model information, please refer to [MediaPipe-Hand-Detection](https://huggingface.co/qualcomm/MediaPipe-Hand-Detection). |
| [sample_resnet101](ai_vision/sample_resnet101)               | QRB ROS Camera      | Y                  | N                   | Y                      | Y                          | The Image Classification is a machine learning model that can classify images from the Imagenet dataset. For model information, please refer to [ResNet101Quantized](https://huggingface.co/qualcomm/ResNet101Quantized). |
| [speech recognition](ai_audio/sample_speech_recognition/)    | N                    | N                  | Y                   | Y                      | Y                          | captures the audio input and publishes the ros topic with the speech recognition result, For model information, please refer to [Whisper-Tiny-En - Qualcomm AI Hub](https://aihub.qualcomm.com/iot/models/whisper_tiny_en?domain=Audio) |
| [speech recognition rt rosnode](ai_audio/sample_speech_recognition_rt_rosnode/)    | USB microphone required | Y                  | Y                   | N                      | N                          | The Speech Recognition Node is a real-time speech-to-text solution via usb mic. It enables developers to integrate voice recognition capabilities into their robotic applications using ROS 2. For model information, please refer to [Whisper-Tiny-En - Qualcomm AI Hub](https://aihub.qualcomm.com/iot/models/whisper_tiny_en?domain=Audio) |
| [sample_object_detection](ai_vision/sample_object_detection/) | QRB ROS Camera          | Y                  | N                   | Y                      | Y                          | The `sample_object_setection` is a Python launch file utilizing QNN for model inference. It demonstrates camera data streaming, AI-based inference, and real-time visualization of object detection results. For model information, please refer to  [YOLOv8-Detection - Qualcomm AI Hub](https://aihub.qualcomm.com/iot/models/yolov8_det?searchTerm=yolov8&domain=Computer+Vision) |
| [sample_object_segmentation](ai_vision/sample_object_segmentation/) | QRB ROS Camera          | Y                  | N                   | Y                      | Y                          | The `sample_object_segmentation` is a Python launch file utilizing QNN for model inference. It demonstrates camera data streaming, AI-based inference, and real-time visualization of object segmentation results.”. For model information, please refer to [YOLOv8-Segmentation - Qualcomm AI Hub](https://aihub.qualcomm.com/iot/models/yolov8_seg?searchTerm=yolov8&domain=Computer+Vision) |
| [sample_hrnet_pose_estimation](ai_vision/sample_hrnet_pose_estimation/) | QRB ROS Camera      | Y                  | N                   | Y                      | N                          | `sample_hrnet_pose_estimation` sample provides high-precision human pose estimation capabilities. For model information, please refer to [HRNetPose - Qualcomm AI Hub](https://aihub.qualcomm.com/iot/models/hrnet_pose?searchTerm=hrnet) |
| [sample_depth_estimation](ai_vision/sample_depth_estimation/) | QRB ROS Camera      | Y                  | N                   | Y                      | N                          | The `sample_depth_estimation` include the pre/post-processs for estimating the depth of each pixel using QNN inference. For model information, please refer to [Depth Anything V2 - Qualcomm AI Hub](https://aihub.qualcomm.com/iot/models/depth_anything_v2?searchTerm=depth&domain=Computer+Vision) |


## List of Robotics Samples

| Sample                                                       | Peripherals required | Develop Ready for Ubuntu | RB3 Gen2 Vision Kit | IQ-9075 Evaluation Kit | IQ-8 Beta   Evaluation Kit | Description                                                  |
| ------------------------------------------------------------ | -------------------- | ------------------------ | ------------------- | ---------------------- | -------------------------- | ------------------------------------------------------------ |
| [simulation_sample_amr_simple_motion](robotics/simulation_sample_amr_simple_motion) | N                    | Y                        | Y                   | Y                      | Y                          | The `AMR simple motion sample` is a Python-based ROS node used to control the simple movements of QRB AMRs within the simulator. This sample allows you to control the movement of QRB AMRs via publishing the ROS messages to `/qrb_robot_base/cmd_vel` topic. |
| [simulation follow me](robotics/simulation_follow_me)                 | N                    | N                        | Y                   | Y                      | Y                          | The `Simulation Follow Me` sample is a AMR to detect, track, and follow a moving person in real time. It integrates sensor emulation and motion control to follow human-following behavior in simulated environments. |
| [simulation_sample_pick_and_place](robotics/simulation_sample_pick_and_place) | N                    | Y                        | Y                   | Y                      | Y                          | The `simulation sample pick and place` is a C++-based robotic manipulation ROS2 node that demonstrates autonomous pick-and-place operations using MoveIt2 for motion planning and Gazebo for physics simulation. |
| [simulation_remote_assistant](robotics/simulation_remote_assistant) | N                    | N                        | Y                   | Y                      | Y                          | The `simulation_remote_assistant` sample application is the ROS package that utilizes an AMR as a remote assistant within a virtual office environment. Users can interact with the robot by inputting natural language commands, such as "Go to the office to check the person." The robot will then autonomously navigate to the specified location and perform object detection tasks as instructed. |
| [simulation_2d_lidar_slam](robotics/simulation_2d_lidar_slam)       | N                    | N                        | Y                   | Y                      | Y                          | The `Simulation 2D Lidar SLAM` sample demonstrates how to run 2D lidar SLAM on Qualcomm robotics platform in a simulated environment, enabling simultaneous localization and mapping.  |
| [simulation_amr_navigation](robotics/simulation_amr_navigation)     | N                    | N                        | Y                   | Y                      | Y                          | The `Simulation AMR Navigation` sample demonstrates how to run navigation2 on Qualcomm robotics platform in a simulated environment.  |
| [sample apriltag](robotics/sample_apriltag) | N                    | Y                        | Y                   | Y                      | Y                          | The `sample_apriltag` is the ROS package to provide AprilTag pipeline samples for Qualcomm robotics platforms. |
| [sample followme](robotics/sample_followme) | orbbec_camera        | Y                        | Y                   | Y                      | Y                          | The `sample_followme` is a AMR to detect, track, and follow a moving person in real time. It integrates sensor emulation and motion control to follow human-following behavior in real environments. This depends on Orbbec Camera to get images |
## System Requirements

- Canonical Ubuntu Image



## Contributions

Thanks for your interest in contributing to qrb ros samples ! Please read our [Contributions Page](CONTRIBUTING.md) for more information on contributing features or bug fixes. We look forward to your participation!

## License

qrb_ros_samples is licensed under the BSD 3-clause "New" or "Revised" License.

Check out the [LICENSE](LICENSE) for more details.
