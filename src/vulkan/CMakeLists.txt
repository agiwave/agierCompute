cmake_minimum_required(VERSION 3.15)
project(ace_be_vulkan VERSION 1.0.0 LANGUAGES C)

# 查找 Vulkan SDK
find_package(Vulkan QUIET)

if(Vulkan_FOUND)
    message(STATUS "Vulkan SDK found: ${Vulkan_VERSION}")
    
    add_library(ace_be_vulkan SHARED vulkan_backend.c)
    
    target_include_directories(ace_be_vulkan PRIVATE 
        ${Vulkan_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    
    target_compile_definitions(ace_be_vulkan PRIVATE 
        ACE_BACKEND_EXPORTS
        VULKAN_AVAILABLE=1
    )
    
    # 查找shaderc库用于SPIRV编译
    find_library(SHADERC_SHARED_LIB NAMES shaderc_shared libshaderc_shared HINTS 
        /lib/x86_64-linux-gnu
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
    )
    
    if(SHADERC_SHARED_LIB)
        message(STATUS "shaderc library found: ${SHADERC_SHARED_LIB}")
        target_compile_definitions(ace_be_vulkan PRIVATE SHADERC_AVAILABLE=1)
        target_link_libraries(ace_be_vulkan PRIVATE ${SHADERC_SHARED_LIB})
    else()
        message(STATUS "shaderc library not found - SPIRV runtime compilation disabled")
    endif()
    
    target_link_libraries(ace_be_vulkan PRIVATE 
        ${Vulkan_LIBRARIES}
        ace_core
    )
    
    # 统一输出目录 - 与示例程序在同一目录
    set_target_properties(ace_be_vulkan PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
    
    install(TARGETS ace_be_vulkan 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
else()
    message(STATUS "Vulkan SDK not found - Vulkan backend disabled")
    add_library(ace_be_vulkan INTERFACE)
endif()