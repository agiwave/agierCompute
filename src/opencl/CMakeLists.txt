# =============================================================================
# AgierCompute OpenCL Backend
# =============================================================================

find_package(OpenCL QUIET)

if(OpenCL_FOUND)
    add_library(ace_be_opencl SHARED
        opencl_backend.c
    )

    target_include_directories(ace_be_opencl
        PRIVATE
            ${OpenCL_INCLUDE_DIRS}
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
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
    message(STATUS "OpenCL SDK not found - OpenCL backend disabled")
    add_library(ace_be_opencl INTERFACE)
endif()
