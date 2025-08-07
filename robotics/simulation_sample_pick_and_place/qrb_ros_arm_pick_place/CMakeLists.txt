cmake_minimum_required(VERSION 3.8)
project(qrb_ros_arm_pick_place)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Create the qrb_ros_arm_pick_place executable
add_executable(qrb_ros_arm_pick_place src/qrb_ros_arm_pick_place.cpp)
target_include_directories(qrb_ros_arm_pick_place PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_features(qrb_ros_arm_pick_place PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  qrb_ros_arm_pick_place
  moveit_msgs
  moveit_ros_planning_interface
  moveit_visual_tools
  rclcpp
  geometry_msgs
  trajectory_msgs
  std_msgs
)

# add_executable(plan_around_objects src/plan_around_objects.cpp)
# target_include_directories(plan_around_objects PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
# target_compile_features(plan_around_objects PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
# ament_target_dependencies(
#   plan_around_objects
#   "moveit_msgs"
#   "moveit_ros_planning_interface"
#   "moveit_visual_tools"
#   "rclcpp"
# )

install(TARGETS
  qrb_ros_arm_pick_place
  DESTINATION lib/${PROJECT_NAME})

# Install launch files if they exist
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.launch.py"
)

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