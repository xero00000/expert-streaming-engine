cmake_minimum_required(VERSION 3.22.1)
project("llama-android")

# Android currently builds the fork itself as a subproject. Keep CUDA/Metal off;
# Vulkan can be enabled by Gradle when a host Vulkan SDK/glslc is available.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(../../../../../../ build-llama)

add_library(${CMAKE_PROJECT_NAME} SHARED
        llama-android.cpp
        expert-android-jni.cpp
        qnn-probe.cpp)

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_17)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        common
        android
        log
        dl)
