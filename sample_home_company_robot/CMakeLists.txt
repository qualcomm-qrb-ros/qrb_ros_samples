cmake_minimum_required(VERSION 3.8)
project(sample_home_company_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake        REQUIRED)
find_package(rclcpp             REQUIRED)
find_package(std_msgs           REQUIRED)
find_package(behaviortree_cpp   REQUIRED)
find_package(ament_index_cpp    REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(follow_me REQUIRED)

# ── Generate custom service ──────────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/FollowMeStart.srv"
  DEPENDENCIES std_msgs
)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(bt_ros_node
  src/main.cpp
  src/receive_whisper_text.cpp
  src/check_follow_me.cpp
  src/send_follow_me_service.cpp
  src/start_fall_detection.cpp
  src/send_to_llm.cpp
  src/send_to_tts.cpp
)

target_include_directories(bt_ros_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(bt_ros_node
  rclcpp
  std_msgs
  behaviortree_cpp
  ament_index_cpp
  follow_me
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(bt_ros_node "${cpp_typesupport_target}")

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS bt_ros_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/  DESTINATION share/${PROJECT_NAME}/launch)

ament_package()
