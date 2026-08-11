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

option(EXPERT_ANDROID_QNN "Build the Qualcomm QAIRT/QNN HTP ggml backend" OFF)
set(QNN_SDK_ROOT "" CACHE PATH "Path to the Qualcomm QAIRT/QNN SDK root")

add_subdirectory(../../../../../../ build-llama)

if(EXPERT_ANDROID_QNN)
    if(NOT QNN_SDK_ROOT AND DEFINED ENV{QNN_SDK_ROOT})
        set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}")
    endif()

    if(NOT EXISTS "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h")
        message(FATAL_ERROR
            "EXPERT_ANDROID_QNN=ON requires QAIRT/QNN. Set QNN_SDK_ROOT to a SDK containing include/QNN/QnnInterface.h")
    endif()

    # Keep QNN Android-specific for now rather than changing every desktop build.
    # The backend uses dlopen(libQnnHtp.so), so only the headers are needed while
    # linking the APK; the matching QAIRT Android runtime is packaged separately.
    target_sources(ggml PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../../../../../../ggml/src/ggml-qnn.cpp")
    target_include_directories(ggml PRIVATE
        "${QNN_SDK_ROOT}/include/QNN")
    target_compile_definitions(ggml PUBLIC GGML_USE_QNN=1)
    target_link_libraries(ggml PRIVATE dl)
endif()

add_library(${CMAKE_PROJECT_NAME} SHARED
        llama-android.cpp
        expert-android-jni.cpp
        qnn-probe.cpp)

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_17)

if(EXPERT_ANDROID_QNN)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE GGML_USE_QNN=1)
endif()

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        common
        android
        log
        dl)
