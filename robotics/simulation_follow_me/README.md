# Simulation Follow Me

The `Simulation Follow Me` sample is a AMR to detect, track, and follow a moving person in real time. It integrates sensor emulation and motion control to  follow human-following behavior in simulated environments.

For more information, please refer to  https://github.com/qualcomm-qrb-ros/qrb_ros_samples/tree/main/ai_vision/sample_object_detction)

<img src="./resource/simulation-followme.gif" style="zoom:80%;" />

## Pipeline flow for Simulation Follow Me



![](./resource/pipeline.png)

## ROS Nodes Used in Simulation Follow Me

## ROS Topics Used in Simulation Follow Me

## Use Case on Ubuntu and QCLINUX

<details>
  <summary>Use Case on Ubuntu</summary>

#### Case: Out of box to run sample on ubuntu

Follow bellow steps on device

```
(ssh) wget https://raw.githubusercontent.com/qualcomm-qrb-ros/qrb_ros_samples/refs/heads/main/tools/qirp-setup.sh -O qirp-setup.sh
(ssh) source qirp-setup.sh

```

</details>

<details>
  <summary> Use cases on QCLINUX</summary>   
#### Prerequisites

- `SSH` is enabled in 'Permissive' mode with the steps mentioned in [Log in using SSH](https://docs.qualcomm.com/bundle/publicresource/topics/80-70017-254/how_to.html?vproduct=1601111740013072&latest=true#use-ssh).

- Download Robotics image and QIRP SDK from [QC artifacts](https://artifacts.codelinaro.org/ui/native/qli-ci/flashable-binaries/qirpsdk/) or Generate Robotics image and QIRP SDK with [meta-qcom-robotics-sdk/README.md](https://github.com/qualcomm-linux/meta-qcom-robotics-sdk)

- The prebuilt robotics image is flashed, see [Flash image](https://docs.qualcomm.com/bundle/publicresource/topics/80-70017-254/flash_images.html?vproduct=1601111740013072&latest=true)

#### Case1: Out of box to run sample on QCLINUX

​	Follow bellow steps on device

```
#source qirp sdk env
(ssh) mount -o remount rw /usr
(ssh) source /usr/share/qirp-setup.sh -m

```

</details>