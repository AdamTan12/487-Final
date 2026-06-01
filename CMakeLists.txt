cmake_minimum_required(VERSION 3.16)
project(gesture_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV 4 REQUIRED)

# Main app: every .cpp under src/, which includes the real main.cpp wired up
# to the CV pipeline.
file(GLOB GD_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(gesture_demo ${GD_SOURCES})
target_include_directories(gesture_demo
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gesture_demo PRIVATE ${OpenCV_LIBS})

# Harness binary for the display side: the real emoji_overlay.cpp plus a
# mock main in tools/. Reuses no other src/ files. Delete once the CV
# pipeline drives draw_gesture for real.
add_executable(emoji_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/emoji_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emoji_overlay.cpp)
target_include_directories(emoji_demo
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS})
target_link_libraries(emoji_demo PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(gesture_demo PRIVATE /W4 /permissive-)
    target_compile_options(emoji_demo   PRIVATE /W4 /permissive-)
else()
    target_compile_options(gesture_demo PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(emoji_demo   PRIVATE -Wall -Wextra -Wpedantic)
endif()
