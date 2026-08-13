cmake_minimum_required(VERSION 3.24)
project(graphPerfScaling LANGUAGES CXX HIP)

set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCE_FILES
    src/graphPerfScaling.hip
    src/graphs.hip)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE include)
