

<div >
  <h1>AI Sample Depth Estimation</h1>
  <p align="center">
</div>

![](./resource/depth_result.gif)

---

## 👋 Overview

- This sample allows you to input an RGB image named `input_image.jpg` or subscribe to the ROS topic `/cam0_stream1` from `qrb ros camera`. It then uses QNN to perform model inference and publishes the result as the `/depth_map` ROS topic containing per-pixel depth values.
- The model is sourced from [Depth Anything V2](https://aihub.qualcomm.com/iot/models/depth_anything_v2?searchTerm=depth&domain=Computer+Vision), a deep convolutional neural network model for depth estimation.

![image-20250723181610392](./resource/depth_estimation_architecture.jpg)

| Node Name                                                    | Function                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [qrb ros camera](https://github.com/qualcomm-qrb-ros/qrb_ros_camera) | Qualcomm ROS 2 package that captures images with parameters and publishes them to ROS topics. |
| image publisher                                              | Publishes image data to a ROS topic—can be camera frames, local files, or processed outputs. |
| sample depth estimation    | Subscribes to input images for preprocessing, then performs postprocessing on the output tensor published by the qrb ros nn interface node. |
| [qrb ros nn interface](https://github.com/qualcomm-qrb-ros/qrb_ros_nn_inference) | Loads a trained AI model, receives preprocessed images, performs inference, and publishes results. |

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

| ROS Topic                       | Type                                          | Description                    |
| ------------------------------- | --------------------------------------------- | ------------------------------ |
| `/image_raw `                   | `<sensor_msgs.msg.Image> `                   | Published image information              |
| `/qrb_inference_input_tensor `  | `<qrb_ros_tensor_list_msgs.msg.TensorList> ` | Preprocessed message             |
| `/qrb_inference_output_tensor ` | `<qrb_ros_tensor_list_msgs.msg.TensorList> ` | Neural network interface result with model |
| `/depth_map ` | `<sensor_msgs.msg.Image> ` | Depth map result              |

## 🎯 Supported targets

<table>
  <tr>
    <th>Development Hardware</th>
    <th>Hardware Overview</th>
  </tr>
  <tr>
    <td>Qualcomm Dragonwing™ IQ-9075 EVK</td>
    <td>
      <a href="https://www.qualcomm.com/products/internet-of-things/industrial-processors/iq9-series/iq-9075">
        <img src="https://s7d1.scene7.com/is/image/dmqualcommprod/dragonwing-IQ-9075-EVK?$QC_Responsive$&fmt=png-alpha" width="160">
      </a>
    </td>
  </tr>
  <tr>
    <td>GMSL Camera Support</td>
    <td>LI-VENUS-OX03F10-OAX40-GM2A-118H(YUV)</td>
  </tr>
</table>

## ✨ Installation

> [!IMPORTANT]
> The following steps need to be run on **Qualcomm Ubuntu** and **ROS Jazzy**.<br>
> Refer to [Install Ubuntu on Qualcomm IoT Platforms](https://ubuntu.com/download/qualcomm-iot) and [Install ROS Jazzy](https://docs.ros.org/en/jazzy/index.html) to setup environment. <br>
> For Qualcomm Linux, please check out the [Qualcomm Intelligent Robotics Product SDK](https://docs.qualcomm.com/bundle/publicresource/topics/80-70018-265/introduction_1.html?vproduct=1601111740013072&version=1.4&facet=Qualcomm%20Intelligent%20Robotics%20Product%20(QIRP)%20SDK) documents.


## 🚀 Usage

<details>
  <summary>Usage details</summary>

## 👨‍💻 Build from source

- Download the Depth Anything V2 model:
```bash
sudo mkdir -p /opt/model && cd /opt/model
sudo wget https://huggingface.co/qualcomm/Depth-Anything-V2/resolve/19ce3645e11de17eed7e869eebcc07dd352834f3/Depth-Anything-V2.bin?download=true -O Depth-Anything-V2.bin
```

- Add qcom ppa repository source:
```bash
sudo add-apt-repository ppa:ubuntu-qcom-iot/qcom-ppa
sudo add-apt-repository ppa:ubuntu-qcom-iot/qirp
sudo apt update
```

- Install dependencies:
```bash
sudo apt install -y ros-dev-tools
sudo apt install -y ros-jazzy-qrb-ros-camera
``` 

- Download source code from the qrb-ros-sample repository:
```bash
mkdir -p ~/qrb_ros_sample_ws/src && cd ~/qrb_ros_sample_ws/src
git clone -b jazzy-rel https://github.com/qualcomm-qrb-ros/qrb_ros_samples.git
```

- Build the sample from source code:
```bash
cd ~/qrb_ros_sample_ws/src/qrb_ros_samples/ai_vision/sample_depth_estimation
rosdep install -i --from-path ./ --rosdistro jazzy -y
colcon build
```

- Source environment and launch demo:
```bash
source install/setup.bash
ros2 launch sample_depth_estimation launch_with_image_publisher.py
```
- You can replace this with a custom image file or model path:
```bash
source install/setup.bash
ros2 launch sample_depth_estimation launch_with_image_publisher.py image_path:=<your local image path> model_path:=<your local model path>
```
- You can also launch with qrb ros camera if you connect the GMSL camera:
```bash
source install/setup.bash
ros2 launch sample_depth_estimation launch_with_qrb_ros_camera.py
```

- When using this launch script, it will use the default parameters, this will send the local `input_image.jpg` file with a publishing rate of 10 Hz. 

```py
    image_path_arg = DeclareLaunchArgument(
        'image_path',
        default_value=os.path.join(package_path, "resource", "input_image.jpg"),
        description='Path to the input image file'
    )

    # Node for image_publisher
    image_publisher_node = Node(
        package='image_publisher',  
        executable='image_publisher_node', 
        namespace=namespace,
        name='image_publisher_node', 
        output='screen', 
        parameters=[
            {'filename': image_path},  
            {'rate': 10.0},  # Set the publishing rate to 10 Hz
        ]
    )
```

- You can then check ROS topics with the topic name `/depth_map` in RViz. 
Please refer to the [ROS 2 Jazzy documentation](https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Introducing-Turtlesim/Introducing-Turtlesim.html) to install rqt.

</details>

## 🤝 Contributing

We love community contributions! Get started by reading our [CONTRIBUTING.md](CONTRIBUTING.md).<br>
Feel free to create an issue for bug reports, feature requests, or any discussion💡.

## ❤️ Contributors

Thanks to all our contributors who have helped make this project better!

<table>
  <tr>
    <td style="text-align: center;">
      <a href="https://github.com/DotaIsMind">
        <img src="https://github.com/DotaIsMind.png" width="100" height="100" alt="teng"/>
        <br />
        <sub><b>teng</b></sub>
      </a>
    </td>
  </tr>
</table>


## ❔ FAQs

<details>
<summary>How to get the original output of the QNN inference node?</summary><br>
Comment out the following code in depth_estimation_node.py to get the original output of the QNN inference node:

```python
# Normalize to [0,255]
normalized = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX)
colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
```
</details>


## 📜 License

Project is licensed under the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) License. See [LICENSE](../../LICENSE) for the full license text.