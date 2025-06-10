cmake_minimum_required(VERSION 3.10)
project(deploy_mujoco)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必要的包
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

# 添加 TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS
    ${CUDA_TOOLKIT_ROOT_DIR}/../TensorRT/include
    /usr/include/x86_64-linux-gnu
    /usr/local/cuda/include
    /usr/local/TensorRT/include
)

find_library(TENSORRT_LIBRARY nvinfer
    PATHS
    ${CUDA_TOOLKIT_ROOT_DIR}/../TensorRT/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
    /usr/local/TensorRT/lib
)

# 添加可执行文件
add_executable(deploy_mujoco deploy_mujoco_tensorrt.cpp)

# 链接库
target_link_libraries(deploy_mujoco
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
    yaml-cpp
)

# 包含目录
target_include_directories(deploy_mujoco PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${YAML_CPP_INCLUDE_DIR}
)
