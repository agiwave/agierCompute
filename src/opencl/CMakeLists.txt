# =============================================================================
# AgierCompute OpenCL Backend
# =============================================================================

include(CheckLanguage)
check_language(OpenCL)

find_package(OpenCL QUIET)

if(OpenCL_FOUND)
    add_library(ace_be_opencl SHARED
        opencl_backend.c
        opencl_device.c
        opencl_memory.c
        opencl_kernel.c
        opencl_type_utils.c
        opencl_dtype_table.c
        opencl_kernel_ops_table.c
    )

    target_include_directories(ace_be_opencl
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${OpenCL_INCLUDE_DIRS}
    )

    target_compile_definitions(ace_be_opencl PRIVATE
        ACE_BACKEND_EXPORTS
        OPENCL_AVAILABLE=1
    )

    target_link_libraries(ace_be_opencl PRIVATE
        ${OpenCL_LIBRARIES}
        ace_core
    )

    set_target_properties(ace_be_opencl PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        OUTPUT_NAME ace_be_opencl
        EXPORT_NAME ace_be_opencl
    )

    # 安装
    install(TARGETS ace_be_opencl
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
else()
    message(STATUS "OpenCL not found - OpenCL backend disabled")
    add_library(ace_be_opencl INTERFACE)
endif()
