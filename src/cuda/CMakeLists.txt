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
        cuda_dtype_table.c
        cuda_kernel_ops_table.c
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

    # 使用 find_package 自动检测 CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    
    # NVRTC 库
    if(TARGET CUDA::nvrtc)
        target_compile_definitions(ace_be_cuda PRIVATE NVRTC_AVAILABLE=1)
        target_link_libraries(ace_be_cuda PRIVATE CUDA::nvrtc)
    elseif(CUDA_nvrtc_LIBRARY)
        target_compile_definitions(ace_be_cuda PRIVATE NVRTC_AVAILABLE=1)
        target_link_libraries(ace_be_cuda PRIVATE ${CUDA_nvrtc_LIBRARY})
    else()
        # 回退：尝试在 CUDA_TOOLKIT_ROOT_DIR 中查找
        find_library(NVRTC_LIB nvrtc
            HINTS
                "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
                "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
                "${CUDA_TOOLKIT_ROOT_DIR}/lib"
        )
        if(NVRTC_LIB)
            target_compile_definitions(ace_be_cuda PRIVATE NVRTC_AVAILABLE=1)
            target_link_libraries(ace_be_cuda PRIVATE ${NVRTC_LIB})
        else()
            message(WARNING "NVRTC library not found - runtime compilation disabled")
        endif()
    endif()

    # CUDA driver API
    if(TARGET CUDA::cuda_driver)
        target_link_libraries(ace_be_cuda PRIVATE CUDA::cuda_driver)
    elseif(CUDA_cuda_driver_LIBRARY)
        target_link_libraries(ace_be_cuda PRIVATE ${CUDA_cuda_driver_LIBRARY})
    else()
        # 回退：尝试直接链接
        if(WIN32)
            find_library(CUDA_DRIVER_LIB cuda
                HINTS
                    "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
                    "${CUDA_TOOLKIT_ROOT_DIR}/lib"
            )
            if(CUDA_DRIVER_LIB)
                target_link_libraries(ace_be_cuda PRIVATE ${CUDA_DRIVER_LIB})
            endif()
        else()
            target_link_libraries(ace_be_cuda PRIVATE cuda)
        endif()
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
    
    message(STATUS "CUDA Toolkit found: ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message(STATUS "CUDA compiler not found - CUDA backend disabled")
    add_library(ace_be_cuda INTERFACE)
endif()
