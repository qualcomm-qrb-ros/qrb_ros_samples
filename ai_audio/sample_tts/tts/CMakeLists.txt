cmake_minimum_required(VERSION 3.14)
project(tts VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PULSEAUDIO REQUIRED libpulse-simple)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/config/tts_paths.cfg" _cfg_lines)
set(QNN_HEADERS "")
foreach(_line IN LISTS _cfg_lines)
    if(_line MATCHES "^[ \t]*qnn_headers_path[ \t]*=[ \t]*(.+)$")
        string(STRIP "${CMAKE_MATCH_1}" QNN_HEADERS)
        break()
    endif()
endforeach()

if(QNN_HEADERS STREQUAL "")
    message(FATAL_ERROR "qnn_headers_path not set in config/tts_paths.cfg")
endif()

message(STATUS "tts: QNN inference enabled (headers=${QNN_HEADERS})")

add_library(tts STATIC
    src/text_normalizer.cpp
    src/sentence_split.cpp
    src/dict_en.cpp
    src/synthesizer.cpp
    src/config_loader.cpp
    src/bert.cpp
    src/inference_engine.cpp
    src/IOTensor.cpp
    src/DataUtil.cpp
)

set_target_properties(tts PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(tts PUBLIC include ${PULSEAUDIO_INCLUDE_DIRS})

target_compile_options(tts PRIVATE
    -Wall -Wextra -Wno-unused-parameter
)

target_compile_definitions(tts PRIVATE
    TTS_DEFAULT_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/tts_paths.cfg"
)

target_include_directories(tts PRIVATE ${QNN_HEADERS})
target_link_libraries(tts PRIVATE dl ${PULSEAUDIO_LIBRARIES})

install(FILES config/tts_paths.cfg DESTINATION share/tts/config)
