cmake_minimum_required(VERSION 2.8)

option(USE_ARM_HPC "Use Arm C/C++ Compiler for HPC" OFF)

if(USE_ARM_HPC)

    set(CMAKE_SYSTEM_NAME Linux)

    set(CMAKE_FIND_ROOT_PATH "/opt/arm/arm-hpc-compiler-1.4_Generic-AArch64_Ubuntu-16.04_aarch64-linux")
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

    set(CMAKE_C_COMPILER "${CMAKE_FIND_ROOT_PATH}/bin/armclang")
    set(CMAKE_CXX_COMPILER "${CMAKE_FIND_ROOT_PATH}/bin/armclang++")

    set(CMAKE_CXX_FLAGS "-mcpu=native ${CMAKE_CXX_FLAGS}")

endif()

project(CNN_Inference_for_ARMv8)

set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -DCPU_ONLY ${CMAKE_CXX_FLAGS}")

set(CAFFE_ROOT "/home/firefly/Documents/caffe-1.0")

include_directories(
    .
    tensor
    layers
    high-level
    utils
    ${CAFFE_ROOT}/include
    ${CAFFE_ROOT}/.build_release/src
)

link_directories(
    ${CAFFE_ROOT}/build/lib
)

add_executable(ZynqNet_Demo example/ZynqNet_Demo.cpp)

target_link_libraries(
    ZynqNet_Demo
    libcaffe.so
    libprotobuf.so
    libglog.so
    libboost_system.so
    libopencv_core.so
    libopencv_imgcodecs.so
    libopencv_imgproc.so
)
