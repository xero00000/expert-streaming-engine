set(TARGET deepseek-v41-eval)
add_executable(${TARGET} deepseek-v41-eval.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
