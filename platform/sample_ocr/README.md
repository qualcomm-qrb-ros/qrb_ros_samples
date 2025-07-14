# OCR_Service User Guide
Here comes the User Guide for OCR Service in QIRP SDK, it will contain some detail show as below.

- the service definition and it's output and input.
- how to build this project.
- how to push binary to device.
- how to run this project.

## Service
### Client

| input                                 | output                                                        |
| ------------------------------------- | ------------------------------------------------------------- |
| Topic name : image topic name(String) | Topic name : OCR result (String) <br>OCR request status: Bool |

### Server and Process Node：

| Input                     | Output                 |
| ------------------------- | ---------------------- |
| Topic : sensor_msgs/Image | Topic : Std_msg/String |

### Test Node:

| Input                     | Output                                              |
| ------------------------- | --------------------------------------------------- |
| String : image topic name <br> String : picture path |  Topic : sensor_msgs/Image |
## Service msg

### ocr_service::ocr_msg::OcrRequest

```
string image_topic_name
---
string ocr_topic_name
bool success
```

Build OCR_Service with QIRP SDK:
```bash

cd <QIRP_SDK>
source setup.sh

cd qirp-sdk/qirp-samples/demos/platform/
cd ocr-service
 
colcon build --merge-install --cmake-args \
  -DPython3_NumPy_INCLUDE_DIR=${Python3_NumPy_INCLUDE_DIR} \
  -DPYTHON_SOABI=cpython-312-aarch64-linux-gnu \
  -DCMAKE_STAGING_PREFIX=$(pwd)/install \
  -DCMAKE_PREFIX_PATH=$(pwd)/install/share \
  -DBUILD_TESTING=OFF \
  -DCMAKE_MAKE_PROGRAM=/usr/bin/make \
  -DPython3_NumPy_INCLUDE_DIR=${OECORE_NATIVE_SYSROOT}/usr/lib/python3.12/site-packages/numpy/core/include

```

## Push
note : Please make sure QIRP is installed on device before you push the OCR ROS Node to device
Push OCR_msg OCR_Service and test picture to device:

```bash
cd install
tar -zcvf ocr_service.tar.gz include/ lib/ share/
ssh root@[ip-addr]
(ssh) mount -o remount rw /
scp ocr_service.tar.gz root@[ip-addr]:/opt
ssh ssh root@[ip-addr]
(ssh) tar --no-overwrite-dir --no-same-owner -zxf /opt/ocr_service.tar.gz -C /usr/

scp  <your pciture> root@[ip-addr]:/opt
```

## RUN
note : Please make sure QIRP is installed on device before you RUN the OCR ROS Node to device
Run ocr_service and ocr_testnode on device:

```bash
Terminal 1: launch ocr_server
export HOME=/home
source /usr/bin/ros_setup.sh && source /usr/share/qirp-setup.sh
ros2 run ocr_service ocr_server

terminal 2: lanunch ocr_testnode
export HOME=/home
source /opt/ros/humble/setup.sh && source /usr/share/qirp-setup.sh
ros2 run ocr_service ocr_testnode --topic <image_topic> --picture <image_path>

terminal 3:launch ocr_client
export HOME=/home
source /usr/bin/ros_setup.sh && source /usr/share/qirp-setup.sh
ros2 run ocr_service ocr_client  <image_topic>
```

## License

This is licensed under the BSD 3-Clause-Clear “New” or “Revised” License.
