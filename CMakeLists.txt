cmake_minimum_required(VERSION 3.16)

project(gesture_recognition
    VERSION 0.1.0
    DESCRIPTION "Real-time hand gesture recognition (OpenCV + ONNX)"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Single output location for binaries across generators
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV 4 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

find_package(onnxruntime REQUIRED)
message(STATUS "onnxruntime found")

add_subdirectory(app)
