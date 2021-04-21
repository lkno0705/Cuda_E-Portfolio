cmake_minimum_required(VERSION 3.19)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(MatrixAddition CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(MatrixAddition main.cu)

set_target_properties(
        MatrixAddition
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)