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
    
    target_link_libraries(ace_be_vulkan PRIVATE 
        ${Vulkan_LIBRARIES}
        ace_core
    )
    
    set_target_properties(ace_be_vulkan PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>"
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