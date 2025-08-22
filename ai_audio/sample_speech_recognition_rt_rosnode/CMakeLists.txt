cmake_minimum_required(VERSION 3.5)
project(qrb_ros_speech_recognition)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

install(
  PROGRAMS
  src/audio_process.py
  src/qrb_ros_speech_recognition.py
  src/qrb_ros_android_asr.py
  src/whisper_transcription.py
  src/tflite_load_model.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
