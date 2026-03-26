# =============================================================================
# AgierCompute CUDA Backend
# =============================================================================

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    add_library(ace_be_cuda SHARED
        cuda_backend.c
        cuda_device.c
        cuda_memory.c
        cuda_kernel.c
        cuda_type_utils.c
    )

    target_include_directories(ace_be_cuda
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
    )

    target_compile_definitions(ace_be_cuda PRIVATE
        ACE_BACKEND_EXPORTS
        CUDA_AVAILABLE=1
    )

    # 查找 NVRTC 库
    find_library(NVRTC_LIB nvrtc
        HINTS
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/lib/x64"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/lib/x64"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64"
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64"
            "C:/Program Files/NVIDIA Corporation/CUDA/v12.6/lib/x64"
            /lib/x86_64-linux-gnu
            /usr/lib/x86_64-linux-gnu
            /usr/local/cuda/lib64
            /usr/local/cuda/lib
    )

    if(NVRTC_LIB)
        target_compile_definitions(ace_be_cuda PRIVATE NVRTC_AVAILABLE=1)
        target_link_libraries(ace_be_cuda PRIVATE ${NVRTC_LIB})
    endif()

    # 链接 CUDA driver API
    if(WIN32)
        find_library(CUDA_DRIVER_LIB cuda
            PATHS
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/lib/x64"
                "C:/Program Files/NVIDIA Corporation/CUDA/*/lib/x64"
        )
        if(CUDA_DRIVER_LIB)
            target_link_libraries(ace_be_cuda PRIVATE ${CUDA_DRIVER_LIB})
        endif()
    else()
        target_link_libraries(ace_be_cuda PRIVATE cuda)
    endif()

    target_link_libraries(ace_be_cuda PRIVATE ace_core)

    set_target_properties(ace_be_cuda PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        OUTPUT_NAME ace_be_cuda
        EXPORT_NAME ace_be_cuda
    )

    # 安装
    install(TARGETS ace_be_cuda
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
else()
    message(STATUS "CUDA compiler not found - CUDA backend disabled")
    add_library(ace_be_cuda INTERFACE)
endif()
