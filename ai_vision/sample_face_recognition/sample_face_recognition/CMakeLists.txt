cmake_minimum_required(VERSION 3.8)
project(sample_face_recognition)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(OpenCV REQUIRED)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/face_detect.cpp
  src/face_recognition.cpp
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )

rclcpp_components_register_nodes(${PROJECT_NAME}
  "sample_face_recognition::RecognitionNode"
)

install(PROGRAMS scripts/image_pub.py
  DESTINATION lib/${PROJECT_NAME}
)

set(_share_items)

set(_candidate_dirs
  "${CMAKE_CURRENT_SOURCE_DIR}/../resource"
  "${CMAKE_CURRENT_SOURCE_DIR}/../model"
  "${CMAKE_CURRENT_SOURCE_DIR}/launch"
)

foreach(_dir IN LISTS _candidate_dirs)
  if(IS_DIRECTORY "${_dir}")
    if(_dir STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/launch")
      list(APPEND _share_items launch)
    else()
      list(APPEND _share_items "${_dir}")
    endif()
  else()
    message(STATUS "[${PROJECT_NAME}] Skip INSTALL_TO_SHARE: directory not found: ${_dir}")
  endif()
endforeach()

ament_auto_package(INSTALL_TO_SHARE
  ${_share_items}
)
