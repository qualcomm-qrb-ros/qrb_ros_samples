#/bin/bash

# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# current dir

HOST_FILES=()

HOST_LIB_PATH="/usr/lib"
HOST_INCLUDE_PATH="/usr/include"
CONTAINER_LIB_PATH="/usr/local/lib"
CONTAINER_INCLUDE_PATH="/usr/local/include"

# libs and head flies of QNN SDK
HOST_FILES+=("-v /usr/bin/qtld-net-run:/usr/bin/qtld-net-run")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libcalculator.so:${CONTAINER_LIB_PATH}/libcalculator.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libhta_hexagon_runtime_snpe.so:${CONTAINER_LIB_PATH}/libhta_hexagon_runtime_snpe.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libPlatformValidatorShared.so:${CONTAINER_LIB_PATH}/libPlatformValidatorShared.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnChrometraceProfilingReader.so:${CONTAINER_LIB_PATH}/libQnnChrometraceProfilingReader.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnCpu.so:${CONTAINER_LIB_PATH}/libQnnCpu.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnDsp.so:${CONTAINER_LIB_PATH}/libQnnDsp.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnDspNetRunExtensions.so:${CONTAINER_LIB_PATH}/libQnnDspNetRunExtensions.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnDspV66CalculatorStub.so:${CONTAINER_LIB_PATH}/libQnnDspV66CalculatorStub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnDspV66Stub.so:${CONTAINER_LIB_PATH}/libQnnDspV66Stub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnGenAiTransformer.so:${CONTAINER_LIB_PATH}/libQnnGenAiTransformer.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnGenAiTransformerCpuOpPkg.so:${CONTAINER_LIB_PATH}/libQnnGenAiTransformerCpuOpPkg.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnGenAiTransformerModel.so:${CONTAINER_LIB_PATH}/libQnnGenAiTransformerModel.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnGpu.so:${CONTAINER_LIB_PATH}/libQnnGpu.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnGpuNetRunExtensions.so:${CONTAINER_LIB_PATH}/libQnnGpuNetRunExtensions.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnGpuProfilingReader.so:${CONTAINER_LIB_PATH}/libQnnGpuProfilingReader.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtp.so:${CONTAINER_LIB_PATH}/libQnnHtp.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtpNetRunExtensions.so:${CONTAINER_LIB_PATH}/libQnnHtpNetRunExtensions.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtpOptraceProfilingReader.so:${CONTAINER_LIB_PATH}/libQnnHtpOptraceProfilingReader.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtpPrepare.so:${CONTAINER_LIB_PATH}/libQnnHtpPrepare.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtpProfilingReader.so:${CONTAINER_LIB_PATH}/libQnnHtpProfilingReader.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtpV68CalculatorStub.so:${CONTAINER_LIB_PATH}/libQnnHtpV68CalculatorStub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtpV68Stub.so:${CONTAINER_LIB_PATH}/libQnnHtpV68Stub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtpV69CalculatorStub.so:${CONTAINER_LIB_PATH}/libQnnHtpV69CalculatorStub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtpV69Stub.so:${CONTAINER_LIB_PATH}/libQnnHtpV69Stub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtpV73CalculatorStub.so:${CONTAINER_LIB_PATH}/libQnnHtpV73CalculatorStub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtpV73Stub.so:${CONTAINER_LIB_PATH}/libQnnHtpV73Stub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnHtpV75Stub.so:${CONTAINER_LIB_PATH}/libQnnHtpV75Stub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnJsonProfilingReader.so:${CONTAINER_LIB_PATH}/libQnnJsonProfilingReader.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnSaver.so:${CONTAINER_LIB_PATH}/libQnnSaver.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnSystem.so:${CONTAINER_LIB_PATH}/libQnnSystem.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libQnnTFLiteDelegate.so:${CONTAINER_LIB_PATH}/libQnnTFLiteDelegate.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libSNPE.so:${CONTAINER_LIB_PATH}/libSNPE.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libSnpeDspV66Stub.so:${CONTAINER_LIB_PATH}/libSnpeDspV66Stub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libSnpeHta.so:${CONTAINER_LIB_PATH}/libSnpeHta.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libSnpeHtpPrepare.so:${CONTAINER_LIB_PATH}/libSnpeHtpPrepare.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libSnpeHtpV68CalculatorStub.so:${CONTAINER_LIB_PATH}/libSnpeHtpV68CalculatorStub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libSnpeHtpV68Stub.so:${CONTAINER_LIB_PATH}/libSnpeHtpV68Stub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libSnpeHtpV73CalculatorStub.so:${CONTAINER_LIB_PATH}/libSnpeHtpV73CalculatorStub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libSnpeHtpV73Stub.so:${CONTAINER_LIB_PATH}/libSnpeHtpV73Stub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libSnpeHtpV75CalculatorStub.so:${CONTAINER_LIB_PATH}/libSnpeHtpV75CalculatorStub.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libSnpeHtpV75Stub.so:${CONTAINER_LIB_PATH}/libSnpeHtpV75Stub.so")

# libs for QNN TFLite GPU delegate
HOST_FILES+=("-v ${HOST_LIB_PATH}/libadreno_utils.so:${CONTAINER_LIB_PATH}/libadreno_utils.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libCB.so:${CONTAINER_LIB_PATH}/libCB.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libOpenCL.so:${CONTAINER_LIB_PATH}/libOpenCL.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libOpenCL_adreno.so:${CONTAINER_LIB_PATH}/libOpenCL_adreno.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libdmabufheap.so.0:${CONTAINER_LIB_PATH}/libdmabufheap.so.0")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libdmabufheap.so.0.0.0:${CONTAINER_LIB_PATH}/libdmabufheap.so.0.0.0")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libgsl.so:${CONTAINER_LIB_PATH}/libgsl.so")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libpropertyvault.so.0:${CONTAINER_LIB_PATH}/libpropertyvault.so.0")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libpropertyvault.so.0.0.0:${CONTAINER_LIB_PATH}/libpropertyvault.so.0.0.0")
HOST_FILES+=("-v ${HOST_LIB_PATH}/libllvm-qcom.so:${CONTAINER_LIB_PATH}/libllvm-qcom.so")

HOST_devices+=("--device=/dev/dri/card0")
HOST_devices+=("--device=/dev/dri/renderD128")
HOST_devices+=("--device=/dev/kgsl-3d0")
HOST_devices+=("--device=/dev/dma_heap/system")
HOST_devices+=("--device=/dev/dma_heap/qcom,system")
HOST_devices+=("--device=/dev/fastrpc-cdsp")
HOST_devices+=("--device=/dev/v4l-subdev8")
HOST_devices+=("--device=/dev/video0")
HOST_devices+=("--device=/dev/video33")
if [ -e /dev/video2 ]; then
    HOST_devices+=("--device=/dev/video2")
fi

# Define Docker image and container names
IMAGE_NAME="qirp-docker"
CONTAINER_NAME="qirp-samples-container"

linux_docker_param=$1
if [ "$2" == "--docker_path" ]; then
    echo "loading docker path from $3"
    IMAGE_PATH=$3
else
    IMAGE_PATH="/home/qirp-docker.tar.gz"
fi

config_file="config.yaml"
ROS_DISTRO=jazzy

model=()
model_label=()
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
    # qnn-tools \
    # libqnn-dev \
    # libqnn1 \
    # libsndfile1-dev \
    # libhdf5-dev \
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
    # qcom-sensors-service  \
    # ros-$ROS_DISTRO-qrb-sensor-client  \
    ros-$ROS_DISTRO-qrb-ros-imu  \
)

pip_packages_base=( \
    typing_extensions  \
    pytesseract \
)

qrb_ros_node=( \
)

apt_packages_sample=()
pip_packages=()


#---------main entry point of ubuntu ------------

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
    # check model dir if exit
    if [ -d "$DIR/model" ]; then
        echo "model code dir is $DIR/model "
    else
        echo "no $DIR/model , creat it..."
        mkdir -p "$DIR/model"
        if [ $? -eq 0 ]; then
            echo "$DIR/model successfully create"
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
        #git clone https://github.qualcomm.com/QUIC-QRB-ROS/qrb_ros_samples.git
        scripts_env_setup
        download_qrb_ros_node $try_times
        touch $DIR/env_check
    fi
    read_configuration
    download_model  $try_times
    download_model_label $try_times
    download_depends
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
    export USB_CAMERA_PATH=$(v4l2-ctl --list-devices | grep -A1 "USB Camera" | tail -n1 | tr -d ' \t')
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
    
    #install qnn libs in real ubuntu env, not in docker
    if [ -f /.dockerenv ]; then
        echo "current in docker , not install qnn and ros base"
        pip3 uninstall numpy
    else
        #add qualcomm carmel ppa
        echo "Add qualcomm carmel ppa , wait few minutes..."
        sudo add-apt-repository -y ppa:ubuntu-qcom-iot/qcom-noble-ppa
        sudo add-apt-repository -y ppa:ubuntu-qcom-iot/qirp
        # apt_packages_base+=(\
            # qnn-tools \
            # libqnn-dev \
            # libqnn1 \
        # )
        
        #add ros apt source
        #sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
        #echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null  
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
    fi  

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
    
        
    #install yq tool
    sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_arm64
    if [ $? -eq 0 ]; then
        echo "add yq tool successfully "
    else
        echo "add yq tool fail"
        exit 1
    fi
    
    sudo chmod a+x /usr/local/bin/yq
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
function read_configuration(){
    # find all config file and read apt/pip/ros nodes
    conf=$(find "$DIR" -type f -name $config_file)
    for file in $conf; do
        echo "Reading conf file: $file"
        model+=($(yq eval '.model[]' $file))
        model_label+=($(yq eval '.model_label[]' $file))
        apt_packages+=($(yq eval '.apt[]' $file))
        pip_packages+=($(yq eval '.pip[]' $file))                
    done

    echo "remove the repeat packages"
    model=($(echo "${model[@]}" | tr ' ' '\n' | awk '!seen[$0]++'))
    model_label=($(echo "${model_label[@]}" | tr ' ' '\n' | awk '!seen[$0]++'))
    apt_packages=($(echo "${apt_packages[@]}" | tr ' ' '\n' | awk '!seen[$0]++'))
    pip_packages=($(echo "${pip_packages[@]}" | tr ' ' '\n' | awk '!seen[$0]++'))
    echo "----------------------------------------"
    echo "need to download model ${model[@]}"
    echo "need to download model_label ${model_label[@]}"
    echo "need to apt install ${apt_packages[@]}"
    echo "need to pip install ${pip_packages[@]}"
    echo "----------------------------------------"
}
function download_model(){
    local N=$1
    if [ $((N -1)) == 0 ];then
        echo "try to download_model $try_times times, fail"
        exit 1
    fi
    IFS=','

    if [ ${#model[@]} -ne 0 ]; then
        for smodel in "${model[@]}"; do
            read model_name link <<< $smodel
            if [ -f $DIR/model/$model_name.done ]; then
                echo "$model_name has been download in $DIR/model"
            else
                wget --no-check-certificate -O $DIR/model/$model_name $link
                if [ $? -eq 0 ]; then
                    echo "download $model_name successfully "
                    touch $DIR/model/$model_name.done
                else
                    download_model $((N-1))
                    exit 1
                fi
            fi
        done
    fi
}
function download_model_label(){
    local N=$1
    if [ $((N -1)) == 0 ];then
        echo "try to download_model $try_times times, fail"
        exit 1
    fi
    IFS=','
    if [ ${#model_label[@]} -ne 0 ]; then
        for smodel in "${model_label[@]}"; do
            read label_name link <<< $smodel
            if [ -f $DIR/model/$label_name.done ]; then
                echo "$label_name has been download in $DIR/model"
            else
                wget --no-check-certificate -O $DIR/model/$label_name $link
                if [ $? -eq 0 ]; then
                    echo "download $label_name successfully "
                    touch $DIR/model/$label_name.done
                else
                    download_model_label $((N-1))
                    exit 1
                fi
            fi
        done
    fi   

}
function download_depends(){
    echo "apt install pkgs ${apt_packages[@]} ... "
    #install apt pkgs
    search_item=""
    for package in "${apt_packages[@]}"; do
        if [[ $package == $search_item ]];then
            echo "$package has been install"
        else
            sudo  DEBIAN_FRONTEND=noninteractive apt-get install -y $package   --fix-missing
            if [ $? -eq 0 ]; then
                echo "apt-get install -y  $package successfully "
            else
                echo "apt-get install -y  $package  fail"
                exit 1
            fi
            search_item+=$package
        fi
    done    
    echo "pip install base pkgs ${pip_packages[@]} ... "
    search_item=""
    #install pip pkgs
    for package in "${pip_packages[@]}"; do
        if [[ $package == $search_item ]];then
            echo "$package has been install"
        else
            sudo pip3 install --user --upgrade --force-reinstall $package
            if [ $? -eq 0 ]; then
                echo "pip install $package successfully "
            else
                echo "pip install $package fail"
                exit 1
            fi
            search_item+=$package
        fi
    done
}

#--------setup in qclinux -----------
function docker_check_install_depends(){
    # Check if Docker image exists
    if ! docker images --format "{{.Repository}}" | grep -w ^$IMAGE_NAME$ ;then
        echo "Docker image $IMAGE_NAME not found, loading from $IMAGE_PATH ..."
        if [ -f $IMAGE_PATH ];then
            docker load -i "$IMAGE_PATH"
            if [ $? -eq 0 ]; then
                echo "Docker image successfully loaded."
            else
                echo "Error loading Docker image."
                exit 1
            fi
        else
            echo "no docker image $IMAGE_NAME in $IMAGE_PATH"
            #docker pull --platform=linux/arm64/v8 ros:$ROS_DISTRO
        fi
    else
        echo "Docker image $IMAGE_NAME already exists."
    fi

    # Check if Docker container exists
    if ! docker ps -a --format "{{.Names}}" | grep -w ^$CONTAINER_NAME$; then
        echo "Docker container $CONTAINER_NAME not found, starting..."
        docker run -d -it --rm \
            -e LOCAL_USER_NAME=$(whoami) \
            -e LOCAL_USER_ID=$(id | awk -F "(" '{print $1}' | sed 's/uid=//') \
            -e LOCAL_GROUP_ID=$(id | awk -F "(" '{print $2}' | awk -F " " '{print $NF}' | sed 's/gid=//') \
            -v $Linux_DIR:$Linux_DIR \
            ${HOST_FILES[@]} \
            ${HOST_devices[@]} \
            --network=host \
            --name=$CONTAINER_NAME \
            --security-opt seccomp=unconfined \
            $IMAGE_NAME:latest   
        if [ $? -eq 0 ]; then
            echo "Docker container successfully started."
        else
            echo "Error starting Docker container."
            exit 1
        fi
        
    else
        echo "Docker container $CONTAINER_NAME already exists."
    fi
    
    docker exec -it -u root $CONTAINER_NAME /bin/bash
    if [ $? -eq 0 ]; then
        echo "Docker container successfully started."
    else
        echo "Error starting Docker container."
        exit 1
    fi
}
function linux_env_setup(){
    echo "Linux setup"

    SDK_NAME="QIRP_SDK"

    #common environment variables export
    export PATH=/bin/aarch64-oe-linux-gcc11.2:/usr/bin:/usr/bin/qim/:${PATH}
    export LD_LIBRARY_PATH=/lib/aarch64-oe-linux-gcc11.2:/usr/lib:/usr/lib/qim:/lib:${LD_LIBRARY_PATH}

    #ROS environment variables export
    export AMENT_PREFIX_PATH=/usr:${AMENT_PREFIX_PATH}
    export PYTHONPATH=/usr/lib/python3.10/site-packages:${PYTHONPATH}

    #gst environment variables export
    export GST_PLUGIN_PATH=/usr/lib/qim/gstreamer-1.0:/usr/lib/gstreamer-1.0:${GST_PLUGIN_PATH}
    export GST_PLUGIN_SCANNER=/usr/libexec/qim/gstreamer-1.0/gst-plugin-scanner:/usr/libexec/gstreamer-1.0/gst-plugin-scanner:${GST_PLUGIN_SCANNER}

    #qnn environment variables export
    export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp:${ADSP_LIBRARY_PATH}
    if [[ $linux_docker_param == "docker" ]];then
        echo "building docker image..."
        setenforce 0
        docker_check_install_depends           
    fi 
    
}


#--------------main point
# if (target:ubuntu):
#     apt install <toolchain list>
function main(){
    DIR="/home/ubuntu/qirp-sdk"
    if lsb_release -a 2>/dev/null | grep -q "Ubuntu"; then  
        check_workdir
        ubuntu_setup
    else
        Linux_DIR="/opt/qirp_ws"
        linux_env_setup     
    fi
}
main
