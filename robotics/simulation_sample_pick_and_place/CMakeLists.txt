cmake_minimum_required(VERSION 3.8)
project(simulation_sample_pick_and_place)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set cross-compilation environment if available
message(STATUS "CMAKE_SYSROOT: $ENV{OECORE_TARGET_SYSROOT}")
# Check build environment
if(DEFINED ENV{OECORE_TARGET_SYSROOT})
    message(STATUS "Found QIRP SDK build environment")
    set(BUILD_ENV "QIRP_SDK")
    message(STATUS "=== CMAKE_SYSROOT: $ENV{OECORE_TARGET_SYSROOT} ===")
    # set(CMAKE_SYSROOT "$ENV{OECORE_TARGET_SYSROOT}")
else()
    message(STATUS "Found normal build environment")
    set(BUILD_ENV "NATIVE")
    message(STATUS "=== CMAKE_SYSROOT: $ENV{CMAKE_SYSROOT} ===")
    # set(CMAKE_SYSROOT "")
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rcutils REQUIRED)

find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(std_msgs REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(eigen_stl_containers REQUIRED)

add_executable(qrb_ros_arm_pick_place src/qrb_ros_arm_pick_place.cpp)

# Prefer target-based include for Eigen
target_link_libraries(qrb_ros_arm_pick_place Eigen3::Eigen)

# Let ament propagate include dirs/libs/defs consistently for BOTH native & cross
ament_target_dependencies(qrb_ros_arm_pick_place
  rclcpp
  rcutils
  moveit_ros_planning_interface
  moveit_msgs
  geometry_msgs
  trajectory_msgs
  std_msgs
  eigen_stl_containers
)

# Install targets and directories
install(TARGETS
    qrb_ros_arm_pick_place
    DESTINATION lib/${PROJECT_NAME}
)

# Install directories
set(INSTALL_DIRS
    launch
    config
    scripts
    meshes
    resource
    src
    urdf
)

foreach(DIR ${INSTALL_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${DIR}")
        install(DIRECTORY
            ${DIR}/
            DESTINATION share/${PROJECT_NAME}/${DIR}
        )
    endif()
endforeach()

ament_package()
