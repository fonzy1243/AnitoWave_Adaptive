cmake_minimum_required(VERSION 4.0)

set(CMAKE_C_COMPILER /usr/bin/gcc-14)
set(CMAKE_CXX_COMPILER /usr/bin/g++-14)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-14)


project(AnitoWave_Adaptive CUDA)

set(CMAKE_CUDA_STANDARD 20)

include_directories("${CMAKE_SOURCE_DIR}/include")

file(GLOB_RECURSE sources
        "${CMAKE_SOURCE_DIR}/src/*.c"
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.cu"
)

add_executable(AnitoWave_Adaptive ${sources})

set_property(TARGET AnitoWave_Adaptive PROPERTY CUDA_ARCHITECTURES 61)
set_target_properties(AnitoWave_Adaptive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
