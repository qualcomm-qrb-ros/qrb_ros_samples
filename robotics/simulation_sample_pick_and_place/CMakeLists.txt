cmake_minimum_required(VERSION 3.8)
project(simulation_sample_pick_and_place)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set cross-compilation environment if available
message(STATUS "CMAKE_SYSROOT: $ENV{OECORE_TARGET_SYSROOT}")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(eigen_stl_containers REQUIRED)

# Set link directories for cross-compilation
if(DEFINED ENV{OECORE_TARGET_SYSROOT})
  link_directories("$ENV{OECORE_TARGET_SYSROOT}/usr/lib")
endif()

# Create the qrb_ros_arm_pick_place executable
add_executable(qrb_ros_arm_pick_place src/qrb_ros_arm_pick_place.cpp)

# Link dependencies
ament_target_dependencies(qrb_ros_arm_pick_place
    moveit_msgs
    moveit_ros_planning_interface
    rclcpp
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

install(DIRECTORY
    launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY
    config/
    DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY
    scripts/
    DESTINATION share/${PROJECT_NAME}/scripts
)

install(DIRECTORY
    meshes/
    DESTINATION share/${PROJECT_NAME}/meshes
)

install(DIRECTORY
    resource/
    DESTINATION share/${PROJECT_NAME}/resource
)

install(DIRECTORY
    src/
    DESTINATION share/${PROJECT_NAME}/src
)

install(DIRECTORY
    urdf/
    DESTINATION share/${PROJECT_NAME}/urdf
)

ament_package()
