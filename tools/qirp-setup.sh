#/bin/bash
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
ROS_DISTRO=jazzy
apt_packages_base=( \
    ros-$ROS_DISTRO-ros-base \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-vision-msgs \
    ros-$ROS_DISTRO-sensor-msgs \
    ros-$ROS_DISTRO-usb-cam  \
    ros-$ROS_DISTRO-rclcpp  \
    ros-$ROS_DISTRO-std-msgs \
    ros-$ROS_DISTRO-fastcdr  \
    ros-$ROS_DISTRO-rosidl-default-runtime  \
    ros-$ROS_DISTRO-ocr-msg  \
    ros-$ROS_DISTRO-rclcpp-components  \
    ros-$ROS_DISTRO-rcutils  \
    tesseract-ocr \
    libopencv-dev \
    python3-numpy \
    python3-opencv \
    python-is-python3 \
    python3-pip \
    python3-colcon-common-extensions \
    python3-pandas \
    wget \
    libpulse-dev \
    python3-requests \
    vim \
    unzip \
    v4l-utils \
)
#qrb ros packages
apt_packages_base+=( \
    ros-$ROS_DISTRO-qrb-ros-transport-image-type \
    ros-$ROS_DISTRO-qrb-ros-transport-imu-type \
    ros-$ROS_DISTRO-qrb-ros-transport-point-cloud2-type \
    ros-$ROS_DISTRO-lib-mem-dmabuf \
    ros-$ROS_DISTRO-ocr-msg  \
    ros-$ROS_DISTRO-ocr-service  \
    ros-$ROS_DISTRO-qrb-ros-system-monitor  \
    ros-$ROS_DISTRO-qrb-ros-system-monitor-interfaces  \
    qcom-battery-client  \  
    qcom-battery-service  \
    ros-$ROS_DISTRO-qrb-ros-battery  \
    ros-$ROS_DISTRO-qrb-ros-imu  \
)
pip_packages_base=( \
    typing_extensions  \
    pytesseract \
)
qrb_ros_node=( \
)
function check_workdir(){ 
    # check src dir if exit
    if [ -d "$DIR/src" ]; then
        echo "workdir is $DIR "
        echo "source code dir is $DIR/src "
    else
        echo "no $DIR/src , creat it..."
        mkdir -p "$DIR/src"
        if [ $? -eq 0 ]; then
            echo "$DIR/src successfully create"
        else
            echo "create $DIR fail"
            exit 1
        fi
    fi

    cd $DIR
    echo "change path to $DIR"
}
#--------setup in ubuntu -----------
function ubuntu_setup(){
    try_times=5
    if [ ! -f "$DIR/env_check" ]; then
        scripts_env_setup
        download_qrb_ros_node $try_times
        touch $DIR/env_check
    fi
    #build depends ros nodes
    source /opt/ros/$ROS_DISTRO/setup.sh
    cd $DIR; colcon build --executor sequential
    if [ $? -eq 0 ]; then
        echo "colcon build ros nodes successfully "
    else
        echo "colcon build ros nodes fail"
        exit 1
    fi
    cd -
    setup_env
}
function setup_env(){
    source /opt/ros/$ROS_DISTRO/setup.sh
    if [ -f $DIR/install/setup.bash ]; then
        source $DIR/install/setup.bash
    fi
    #export USB_CAMERA_PATH=$(v4l2-ctl --list-devices | grep -A1 "USB Camera" | tail -n1 | tr -d ' \t')
    if [ $? -eq 0 ]; then
        echo "setup_env successfully "
    else
        echo "setup_env  fail"
        exit 1
    fi
}
function scripts_env_setup(){
    #download depends
    sudo apt update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common curl 
    if [ $? -eq 0 ]; then
        echo "apt-get install -y  software-properties-common curl successfully "
    else
        echo "apt-get install -y  software-properties-common curl  fail"
        exit 1
    fi
    #add qualcomm carmel ppa
    echo "Add qualcomm carmel ppa , wait few minutes..."
    sudo add-apt-repository -y ppa:ubuntu-qcom-iot/qcom-noble-ppa
    sudo add-apt-repository -y ppa:ubuntu-qcom-iot/qirp
    #add ros ppa 
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" # If using Ubuntu derivates use $UBUNTU_CODENAME
    sudo apt install /tmp/ros2-apt-source.deb
    if [ $? -eq 0 ]; then
        echo "add ros base successfully "
    else
        echo "add ros base fail"
        exit 1
    fi
    sudo apt update && sudo apt upgrade -y
    #install apt pkgs
    echo "apt install base pkgs ${apt_packages_base[@]} ... "
    for package in "${apt_packages_base[@]}"; do
        sudo  DEBIAN_FRONTEND=noninteractive apt-get install -y $package   --fix-missing
        if [ $? -eq 0 ]; then
            echo "apt-get install -y  $package successfully "
        else
            echo "apt-get install -y  $package  fail"
            exit 1
        fi
    done
    #can't pip install in ubuntu24.04 docker 
    if [ -f /usr/lib/python3.12/EXTERNALLY-MANAGED ]; then
        sudo mv /usr/lib/python3.12/EXTERNALLY-MANAGED  /usr/lib/python3.12/EXTERNALLY-MANAGED.bk
    fi
    #install pip pkgs
    echo "pip install base pkgs ${pip_packages_base[@]} ... "
    for package in "${pip_packages_base[@]}"; do
        pip install $package
        if [ $? -eq 0 ]; then
            echo "pip install $package successfully "
        else
            echo "pip install $package fail"
            exit 1
        fi  
    done
}
function download_qrb_ros_node(){
    cd $DIR/src
    local N=$1
    if [ $((N -1)) == 0 ];then
        echo "try to git clone $node $try_times times, fail"
        exit 1
    fi
    for node in ${qrb_ros_node[@]};do
        if [ -f $DIR/src/$(echo $node | awk -F'/' '{print $NF}').done ]; then 
            echo "$node has been download in $DIR/src"
        else
            git clone $node
            if [ $? -eq 0 ]; then
                echo "git clone $node successfully "
                touch $DIR/src/$(echo $node | awk -F'/' '{print $NF}').done
            else
                download_qrb_ros_node $((N-1))
                exit 1
            fi
        fi
    done
}
DIR="/home/ubuntu/qirp-sdk"
function main(){
    check_workdir
    ubuntu_setup
}
main
