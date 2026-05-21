cmake_minimum_required(VERSION 3.8)
project(follow_me)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(qrb_ros_people_reid REQUIRED)

# Generate service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/StateControl.srv"
)

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# Person Tracker Node
# add_executable(person_tracker_node
#   src/person_tracker_node.cpp
#   src/template_pool.cpp
#   src/pid_controller.cpp
#   src/depth_processor.cpp
#   src/state_machine.cpp
# )

# ament_target_dependencies(person_tracker_node
#   rclcpp
#   std_msgs
#   sensor_msgs
#   geometry_msgs
#   vision_msgs
#   cv_bridge
#   image_transport
#   message_filters
#   # qrb_ros_people_reid
# )

# target_link_libraries(person_tracker_node
#   ${OpenCV_LIBRARIES}
# )

# rosidl_get_typesupport_target(cpp_typesupport_target
#   ${PROJECT_NAME} rosidl_typesupport_cpp)

# target_link_libraries(person_tracker_node "${cpp_typesupport_target}")

# Install targets
# install(TARGETS
#   person_tracker_node
#   DESTINATION lib/${PROJECT_NAME}
# )

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install header files
install(DIRECTORY
  include/
  DESTINATION include/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
