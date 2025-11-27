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
    set(CMAKE_SYSROOT "$ENV{OECORE_TARGET_SYSROOT}")
else()
    message(STATUS "Found normal build environment")
    set(BUILD_ENV "NATIVE")
    set(CMAKE_SYSROOT "")
endif()

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

include_directories(${EIGEN3_INCLUDE_DIRS}) 
include_directories("$ENV{OECORE_TARGET_SYSROOT}/usr/include") 

# Create the qrb_ros_arm_pick_place executable
add_executable(qrb_ros_arm_pick_place src/qrb_ros_arm_pick_place.cpp)

# Link dependencies
if(BUILD_ENV STREQUAL "NATIVE")
  message(STATUS "Linking native libraries")
  ament_target_dependencies(qrb_ros_arm_pick_place
      moveit_msgs
      moveit_ros_planning_interface
      rclcpp
      geometry_msgs
      trajectory_msgs
      std_msgs
      eigen_stl_containers
  )
endif()

if(BUILD_ENV STREQUAL "QIRP_SDK")
  message(STATUS "set OECORE_TARGET_SYSROOT include path: $ENV{OECORE_TARGET_SYSROOT}/usr/include")
  target_include_directories(qrb_ros_arm_pick_place PUBLIC 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/bullet" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/moveit_ros_planning_interface" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/moveit_core" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/urdf" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/urdfdom_headers" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/moveit_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rosidl_runtime_cpp" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rosidl_runtime_c" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rosidl_typesupport_interface" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rcutils" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/geometry_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rclcpp" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rcl" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rmw" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rcl_yaml_param_parser" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/type_description_interfaces" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/type_description_interfaces" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/service_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/builtin_interfaces" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rcpputils" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rosidl_dynamic_typesupport" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rcl_interfaces" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/tracetools" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rosidl_typesupport_introspection_cpp" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/libstatistics_collector" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/statistics_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/std_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/trajectory_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/sensor_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/visualization_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/object_recognition_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/shape_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/octomap_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/unique_identifier_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/action_msgs" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rclcpp_action" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/rcl_action" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/tf2_ros" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/tf2" 
    "$ENV{OECORE_TARGET_SYSROOT}/usr/include/tf2_msgs" 
    
    ) 
  target_link_directories(qrb_ros_arm_pick_place PUBLIC "$ENV{OECORE_TARGET_SYSROOT}/usr/lib")
  target_link_libraries(qrb_ros_arm_pick_place PUBLIC rclcpp rcutils moveit_move_group_interface)
endif()

# target_link_libraries(qrb_ros_arm_pick_place PUBLIC rclcpp rcutils moveit_move_group_interface)

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
