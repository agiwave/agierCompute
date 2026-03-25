cmake_minimum_required(VERSION 3.15)
project(ace_be_opencl VERSION 1.0.0 LANGUAGES C)

# 查找 OpenCL SDK
find_package(OpenCL QUIET)

if(OpenCL_FOUND)
    message(STATUS "OpenCL SDK found: ${OpenCL_VERSION_STRING}")
    
    add_library(ace_be_opencl SHARED opencl_backend.c)
    
    target_include_directories(ace_be_opencl PRIVATE 
        ${OpenCL_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    
    target_compile_definitions(ace_be_opencl PRIVATE 
        ACE_BACKEND_EXPORTS
        OPENCL_AVAILABLE=1
    )
    
    target_link_libraries(ace_be_opencl PRIVATE 
        ${OpenCL_LIBRARIES}
        ace_core
    )
    
    # 统一输出目录 - 与示例程序在同一目录
    set_target_properties(ace_be_opencl PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
    
    install(TARGETS ace_be_opencl 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
else()
    message(STATUS "OpenCL SDK not found - OpenCL backend disabled")
    add_library(ace_be_opencl INTERFACE)
endif()