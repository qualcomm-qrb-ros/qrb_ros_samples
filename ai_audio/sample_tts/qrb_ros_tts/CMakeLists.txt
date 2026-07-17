cmake_minimum_required(VERSION 3.14)
project(qrb_ros_tts)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../tts
  ${CMAKE_CURRENT_BINARY_DIR}/tts_build
)

# ── Component library ─────────────────────────────────────────────────────────
add_library(speaker_node_component SHARED
  src/speaker_node.cpp
)

target_compile_definitions(speaker_node_component PRIVATE QRB_ROS_TTS_BUILDING_DLL)

target_include_directories(speaker_node_component PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../tts/include
)

ament_target_dependencies(speaker_node_component
  rclcpp
  rclcpp_components
  std_msgs
)

target_link_libraries(speaker_node_component
  tts
)

rclcpp_components_register_nodes(speaker_node_component "SpeakerNode")

# ── Standalone executable ─────────────────────────────────────────────────────
add_executable(speaker_node src/main.cpp)

target_include_directories(speaker_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

ament_target_dependencies(speaker_node rclcpp)

target_link_libraries(speaker_node speaker_node_component)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS speaker_node_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS speaker_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../tts/config/tts_paths.cfg
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
