cmake_minimum_required(VERSION 4.0)
project(AnitoWave_Adaptive CUDA)

set(CMAKE_CUDA_STANDARD 23)

include_directories("${CMAKE_SOURCE_DIR}/include")

file(GLOB_RECURSE sources
        "${CMAKE_SOURCE_DIR}/src/*.c"
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.cu"
)

add_executable(AnitoWave_Adaptive ${sources})

set_target_properties(AnitoWave_Adaptive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
