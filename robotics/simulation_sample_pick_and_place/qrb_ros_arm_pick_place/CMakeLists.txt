cmake_minimum_required(VERSION 3.8)
project(qrb_ros_arm_pick_place)

# set(CMAKE_SYSROOT "$ENV{OECORE_TARGET_SYSROOT}")
message(STATUS "CMAKE_SYSROOT: $ENV{OECORE_TARGET_SYSROOT}")
# set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(eigen_stl_containers REQUIRED)

message(STATUS "Eigen3_INCLUDE_DIRS: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "eigen_stl_containers_INCLUDE_DIRS: ${eigen_stl_containers_INCLUDE_DIRS}")

include_directories(${EIGEN3_INCLUDE_DIRS})

# Create the qrb_ros_arm_pick_place executable
add_executable(qrb_ros_arm_pick_place src/qrb_ros_arm_pick_place.cpp)

link_directories("${OECORE_TARGET_SYSROOT}/usr/lib")

target_link_libraries(qrb_ros_arm_pick_place
    "$ENV{OECORE_TARGET_SYSROOT}/usr/lib/libm.so"
)

ament_target_dependencies(${PROJECT_NAME} 
    moveit_msgs
    moveit_ros_planning_interface
    rclcpp
    geometry_msgs
    trajectory_msgs
    std_msgs
    eigen_stl_containers
)

install(TARGETS
    qrb_ros_arm_pick_place
    DESTINATION lib/${PROJECT_NAME})

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
