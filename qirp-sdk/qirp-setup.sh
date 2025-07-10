#/bin/bash

# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# current dir

ROS_DISTRO=jazzy

#scripts base ros packages
apt_packages_base=( \
)

pip_packages_base=( \
    typing_extensions  \
    pytesseract \
)

qrb_ros_node=( \
)

DIR=/home/ubuntu

#--------setup in ubuntu -----------
function ubuntu_setup(){
    try_times=5     
    scripts_env_setup
    download_qrb_ros_node $try_times
    setup_env
}
function setup_env(){
    source /opt/ros/$ROS_DISTRO/setup.sh
    export USB_CAMERA_PATH=$(v4l2-ctl --list-devices | grep -A1 "USB Camera" | tail -n1 | tr -d ' \t')
}
function scripts_env_setup(){ 
    echo "apt install base pkgs ${apt_packages_base[@]} ... "
    #install apt pkgs
    for package in "${apt_packages_base[@]}"; do
        sudo  DEBIAN_FRONTEND=noninteractive apt-get install -y $package   --fix-missing
        if [ $? -eq 0 ]; then
            echo "apt-get install -y  $package successfully "
        else
            echo "apt-get install -y  $package  fail"
            exit 1
        fi
    done
    
    echo "pip install base pkgs ${pip_packages_base[@]} ... "
    #can't pip install in ubuntu24.04 docker 
    if [ -f /usr/lib/python3.12/EXTERNALLY-MANAGED ]; then
        sudo mv /usr/lib/python3.12/EXTERNALLY-MANAGED  /usr/lib/python3.12/EXTERNALLY-MANAGED.bk
    fi
    
    #install pip pkgs
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
    if [ -d "$DIR/src" ]; then
       echo "✅ Directory exists: $DIR/src"
    else
       echo "⚠️ Directory not found: $DIR/src"
       mkdir -p "$DIR/src"
    fi
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
    cd -
}

#--------------main point--------------#
main(){
    ubuntu_setup
    if [[ $? -eq 0 ]]; then
        echo "Setup qirp sdk successfully!"
     else
        echo "Something went wrong. Please check error info."
        return 1
     fi
}
main
