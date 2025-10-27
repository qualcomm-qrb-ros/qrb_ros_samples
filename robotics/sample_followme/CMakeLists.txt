cmake_minimum_required(VERSION 3.10)
project(follow_me)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
option(BUILD_WITH_OPENMP "Use OpenMP" ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
# find_package(rcl_logging_noop REQUIRED)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

set(BOOST_THREAD_LIBRARY boost_thread -pthread)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/tracker.cpp
    src/controller.cpp
)

target_link_libraries(${PROJECT_NAME}
    # realsense2
    ncnn
    pthread dl
    ${OpenCV_LIBS}
)

set(dependencies
    rclcpp
    geometry_msgs
    sensor_msgs
    message_filters
    nav_msgs
    rclcpp_components
    std_msgs
    tf2
    tf2_ros
    image_transport
    cv_bridge
    # rcl_logging_noop
    ncnn
)

ament_target_dependencies(${PROJECT_NAME}
    ${dependencies}
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies})

ament_package()

