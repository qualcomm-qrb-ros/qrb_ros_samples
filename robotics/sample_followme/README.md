<div align="center">
  <h1>Follow Me</h1>
  <!-- <p align="center">
    <img src='https://via.placeholder.com/600x400?text=Person+Tracking+Demo' alt='Follow Me Demo'></img>
  </p> -->
  <p>Person tracking and following system based on Re-ID with external service integration for ROS 2</p>

  <a href="https://docs.ros.org/en/jazzy/" target="_blank"><img src="https://img.shields.io/badge/ROS%202-Jazzy-1c428a?style=for-the-badge&logo=ros&logoColor=white" alt="ROS 2 Jazzy"></a>

</div>

---

## 👋 Overview

The `follow_me` is the ROS 2 package for person tracking and following using Re-ID features. It integrates with an external Re-ID service package to extract features and compute similarity scores for robust person tracking in multi-person scenarios.

<!-- <div align="center">
  <img src="./resource/architecture.png" alt="architecture" width="600">
</div> -->

<br>

- The pipeline accepts RGB images from `/camera/color/image_raw`, depth images from `/camera/depth/image_raw`, and person detections from `/yolo_detect_result`.
- Target initialization occurs when exactly one person is detected in the image center within the specified distance range (1.5m - 3.0m).
- Re-ID matching uses external services: `ExtractFeature` for feature extraction and `ComputeSimilarity` for similarity computation.
- Matching logic: similarity < match_threshold means same person; similarity < template_update_threshold triggers template update.
- The node publishes control commands to `/cmd_vel` to keep the target centered and at target distance.

## 🔎 Table of contents

  * [Used ROS Topics](#-used-ros-topics)
  * [Re-ID Service Interface](#-re-id-service-interface)
  * [Installation](#-installation)
  * [Usage](#-usage)
  * [Build from source](#-build-from-source)
  * [Contributing](#-contributing)
  * [License](#-license)

## ⚓ Used ROS Topics

| ROS Topic                       | Type                                          | Description                    |
| ------------------------------- | --------------------------------------------- | ------------------------------ |
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | RGB camera stream |
| `/camera/depth/image_raw` | `sensor_msgs/msg/Image` | Depth image stream |
| `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | Camera intrinsics |
| `/yolo_detect_result` | `vision_msgs/msg/Detection2DArray` | Person detections from YOLO |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Robot velocity commands |
| `/target_visualization` | `sensor_msgs/msg/Image` | Debug visualization with bounding boxes |
| `/follow_me/state_control` | `follow_me/srv/StateControl` | State control service (start/pause/resume/finish) |

## ✨ Re-ID Service Interface

This project requires an external Re-ID package that provides the following services:

| Service | Request | Response | Description |
| --- | --- | --- | --- |
| `/extract_feature` | `sensor_msgs/Image image` | `float32[] feature`, `bool success`, `string message` | Extract feature vector from image |
| `/compute_similarity` | `float32[] feature1`, `float32[] feature2` | `float32 similarity`, `bool success`, `string message` | Compute cosine similarity between two features |


## 🚀 Usage

### Start the simulator on host

Please refer to the `Quick Start` of [QRB ROS Simulation](https://github.com/qualcomm-qrb-ros/qrb_ros_simulation) to launch `QRB Robot Base AMR` on host. Ensure that the device and the host are on the same local network and can communicate with each other via ROS communication.

```bash
ros2 launch qrb_ros_sim_gazebo gazebo_robot_base_mini.launch.py world_model:=warehouse_followme_path2 enable_laser:=false
```

### Build follow_me and re-id on device

```bash
git clone ...
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```
### Start the yolo object detection

Please refer to [sample object detection](https://github.com/qualcomm-qrb-ros/qrb_ros_samples/tree/main/ai_vision/sample_object_detection) to launch `object detection` on device.


```bash
ros2 launch sample_object_detection launch_with_orbbec_camera.py model:=/opt/model/yolov8_det_qcs9075.bin score_thres:=0.4
```

### Start the reid service
```bash
ros2 launch qrb_ros_people_reid qrb_sample_people_reid.launch.py reid_model_path:=xxx/sample_people_reid/osnet.bin
```

### Start the tracking pipeline

```bash
# Launch follow_me
ros2 launch follow_me person_tracking.launch.py use_sim_time:=true

# Launch with debug logging
# ros2 launch follow_me person_tracking.launch.py use_sim_time:=true log_level:=DEBUG
```

### Control the tracker state

Use the `StateControl` service to manage tracking:

```bash
# Start tracking
ros2 service call /follow_me/state_control \
  follow_me/srv/StateControl "{set_state: 1}"

# Pause tracking
ros2 service call /follow_me/state_control \
  follow_me/srv/StateControl "{set_state: 2}"

# Resume tracking
ros2 service call /follow_me/state_control \
  follow_me/srv/StateControl "{set_state: 3}"

# Finish tracking
ros2 service call /follow_me/state_control \
  follow_me/srv/StateControl "{set_state: 4}"
```

### Monitor tracking on host

```bash
# View debug visualization
ros2 run rqt_image_view rqt_image_view /target_visualization
```

## Configuration

Main parameters can be configured via launch arguments or in `config/tracking_params.yaml`:

```yaml
person_tracker_node:
  ros__parameters:
    # Detection parameters
    detection_image_width: 640    # Detection model input image width (pixels)
    detection_image_height: 640   # Detection model input image height (pixels)
    
    # Tracking parameters
    target_distance: 2.0          # Target tracking distance (meters)
    min_init_distance: 1.5        # Minimum initialization distance (meters)
    max_init_distance: 3.0        # Maximum initialization distance (meters)
    person_class_id: 0            # Person class ID in detection (0 for COCO dataset)
    
    # Re-ID parameters
    template_update_threshold: 0.1  # Template update threshold
    match_threshold: 0.3            # Matching threshold
    drift_threshold: 0.6             # Anti-drift threshold
    max_processing_time_ms: 200      # Maximum processing time (milliseconds)
    
    # PID parameters - Linear velocity
    linear_kp: 0.5
    linear_ki: 0.0
    linear_kd: 0.1
    
    # PID parameters - Angular velocity
    angular_kp: 2.0
    angular_ki: 0.0
    angular_kd: 0.2
    
    # Velocity limits
    max_linear_speed: 0.5   # Maximum linear velocity (m/s)
    max_angular_speed: 0.5  # Maximum angular velocity (rad/s)
    
    # Debug mode
    debug_mode: true        # Enable debug mode (publish visualization)
```

## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).<br>
Feel free to create an issue for bug report, feature requests or any discussion💡.

## 📜 License

Project is licensed under the [BSD-3-Clause-Clear](https://spdx.org/licenses/BSD-3-Clause-Clear.html) License. See [LICENSE](../../LICENSE) for the full license text.
