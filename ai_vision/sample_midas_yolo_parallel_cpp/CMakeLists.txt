cmake_minimum_required(VERSION 3.8)
project(sample_midas_yolo_parallel_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # Real-time depth/detection fusion is perf-sensitive; default to optimized
  # builds unless the user explicitly requests Debug.
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O3)
  endif()
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

ament_auto_add_library(midas_yolo_fusion_node SHARED
  src/midas_yolo_fusion_node.cpp
  src/fast_image_publisher_node.cpp)

target_include_directories(midas_yolo_fusion_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(midas_yolo_fusion_node
  ${OpenCV_LIBS})

rclcpp_components_register_node(
  midas_yolo_fusion_node
  PLUGIN "sample_midas_yolo_parallel_cpp::MidasYoloFusionNode"
  EXECUTABLE midas_yolo_fusion_node_exe)

rclcpp_components_register_node(
  midas_yolo_fusion_node
  PLUGIN "sample_midas_yolo_parallel_cpp::FastImagePublisherNode"
  EXECUTABLE fast_image_publisher_node_exe)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
