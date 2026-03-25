cmake_minimum_required(VERSION 3.15)
project(ace_be_cuda VERSION 1.0.0 LANGUAGES C)

# 现代 CMake 方式查找 CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    
    add_library(ace_be_cuda SHARED cuda_backend.c)
    
    target_include_directories(ace_be_cuda PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    
    target_compile_definitions(ace_be_cuda PRIVATE 
        ACE_BACKEND_EXPORTS
        CUDA_AVAILABLE=1
    )
    
    # 查找NVRTC库
    find_library(NVRTC_LIB nvrtc HINTS 
        /lib/x86_64-linux-gnu
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/local/cuda/lib
    )
    
    if(NVRTC_LIB)
        message(STATUS "NVRTC library found: ${NVRTC_LIB}")
        target_compile_definitions(ace_be_cuda PRIVATE NVRTC_AVAILABLE=1)
        target_link_libraries(ace_be_cuda PRIVATE ${NVRTC_LIB})
    else()
        message(STATUS "NVRTC library not found - runtime compilation disabled")
    endif()
    
    # 链接 CUDA driver API
    if(WIN32)
        find_library(CUDA_DRIVER_LIB cuda PATHS 
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/lib/x64"
            "C:/Program Files/NVIDIA Corporation/CUDA/*/lib/x64"
        )
        if(CUDA_DRIVER_LIB)
            message(STATUS "CUDA driver library: ${CUDA_DRIVER_LIB}")
            target_link_libraries(ace_be_cuda PRIVATE ${CUDA_DRIVER_LIB})
        endif()
    else()
        target_link_libraries(ace_be_cuda PRIVATE cuda)
    endif()
    
    target_link_libraries(ace_be_cuda PRIVATE ace_core)
    
    # 统一输出目录 - 与示例程序在同一目录
    set_target_properties(ace_be_cuda PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
    
    install(TARGETS ace_be_cuda 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
else()
    message(STATUS "CUDA compiler not found - CUDA backend disabled")
    add_library(ace_be_cuda INTERFACE)
endif()