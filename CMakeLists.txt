cmake_minimum_required(VERSION 3.14)
project(opencv_project)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(mat src/mat_test.cpp)
add_executable(detection src/detection.cpp)

target_link_libraries(mat PRIVATE ${OpenCV_LIBS})
target_link_libraries(detection PRIVATE ${OpenCV_LIBS})