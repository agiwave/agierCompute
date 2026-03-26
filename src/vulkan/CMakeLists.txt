# =============================================================================
# AgierCompute Vulkan Backend
# =============================================================================

# 设置 Vulkan SDK 路径（Windows 常见安装位置）
if(WIN32)
    set(VULKAN_SDK_PATHS
        "C:/Apps/VulkanSDK"
        "C:/VulkanSDK"
        "$ENV{VULKAN_SDK}"
    )
endif()

find_package(Vulkan QUIET)

if(Vulkan_FOUND)
    add_library(ace_be_vulkan SHARED
        vulkan_backend.c
    )

    target_include_directories(ace_be_vulkan
        PRIVATE
            ${Vulkan_INCLUDE_DIRS}
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
    )

    target_compile_definitions(ace_be_vulkan PRIVATE
        ACE_BACKEND_EXPORTS
        VULKAN_AVAILABLE=1
    )

    # 查找 shaderc 库用于 SPIR-V 编译
    find_library(SHADERC_SHARED_LIB NAMES shaderc_shared shaderc_combined shaderc
        HINTS
            ${VULKAN_SDK_PATHS}
            "${Vulkan_INCLUDE_DIRS}/.."
            "${Vulkan_INCLUDE_DIRS}/../Lib"
            "C:/Apps/VulkanSDK/Lib"
            "C:/VulkanSDK/Lib"
            /lib/x86_64-linux-gnu
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
    )

    if(SHADERC_SHARED_LIB)
        target_compile_definitions(ace_be_vulkan PRIVATE SHADERC_AVAILABLE=1)
        target_link_libraries(ace_be_vulkan PRIVATE ${SHADERC_SHARED_LIB})
        message(STATUS "  Shaderc library found: ${SHADERC_SHARED_LIB}")
    else()
        message(STATUS "  Shaderc library not found - Vulkan will work but cannot compile kernels at runtime")
    endif()

    target_link_libraries(ace_be_vulkan PRIVATE
        ${Vulkan_LIBRARIES}
        ace_core
    )

    set_target_properties(ace_be_vulkan PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        OUTPUT_NAME ace_be_vulkan
        EXPORT_NAME ace_be_vulkan
    )

    # 安装
    install(TARGETS ace_be_vulkan
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
else()
    message(STATUS "Vulkan SDK not found - Vulkan backend disabled")
    add_library(ace_be_vulkan INTERFACE)
endif()
