file(GLOB GR_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(gesture_recognition ${GR_SOURCES})

target_include_directories(gesture_recognition
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS})

target_link_libraries(gesture_recognition PRIVATE ${OpenCV_LIBS} onnxruntime::onnxruntime)

if(MSVC)
    target_compile_options(gesture_recognition PRIVATE /W4 /permissive-)
else()
    target_compile_options(gesture_recognition PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
endif()

set_target_properties(gesture_recognition PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
