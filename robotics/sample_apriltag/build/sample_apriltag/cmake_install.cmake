# Install script for directory: /local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/install/sample_apriltag")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/local/mnt/workspace/fulan/test/qirp-sdk/toolchain/install_dir/sysroots/x86_64-qcomsdk-linux/usr/bin/aarch64-qcom-linux/aarch64-qcom-linux-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag" TYPE DIRECTORY FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/launch")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag" TYPE DIRECTORY FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/config")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/sample_apriltag")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/sample_apriltag")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag/environment" TYPE FILE FILES "/local/mnt/workspace/fulan/test/qirp-sdk/toolchain/install_dir/sysroots/armv8-2a-qcom-linux/usr/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag/environment" TYPE FILE FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag/environment" TYPE FILE FILES "/local/mnt/workspace/fulan/test/qirp-sdk/toolchain/install_dir/sysroots/armv8-2a-qcom-linux/usr/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag/environment" TYPE FILE FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_environment_hooks/path.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag" TYPE FILE FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_environment_hooks/local_setup.bash")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag" TYPE FILE FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_environment_hooks/local_setup.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag" TYPE FILE FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag" TYPE FILE FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag" TYPE FILE FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_environment_hooks/package.dsv")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_index/share/ament_index/resource_index/packages/sample_apriltag")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag/cmake" TYPE FILE FILES
    "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_core/sample_apriltagConfig.cmake"
    "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/ament_cmake_core/sample_apriltagConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/sample_apriltag" TYPE FILE FILES "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/local/mnt/workspace/fulan/demo_templ/qrb_ros_samples_fulan/robotics/sample_apriltag/build/sample_apriltag/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
