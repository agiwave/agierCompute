# =============================================================================
# AgierCompute Vulkan Backend
# =============================================================================

find_package(Vulkan QUIET)

if(Vulkan_FOUND)
    add_library(ace_be_vulkan SHARED
        vulkan_backend.c
        vulkan_device.c
        vulkan_device_features.c
        vulkan_memory.c
        vulkan_kernel.c
        vulkan_type_utils.c
        vulkan_dtype_table.c
        vulkan_kernel_ops_table.c
    )

    target_include_directories(ace_be_vulkan
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${Vulkan_INCLUDE_DIRS}
    )

    target_compile_definitions(ace_be_vulkan PRIVATE
        ACE_BACKEND_EXPORTS
        VULKAN_AVAILABLE=1
    )

    # 查找 shaderc
    find_library(SHADERC_LIB shaderc_shared
        HINTS
            /usr/lib
            /usr/local/lib
            "C:/VulkanSDK/*/Lib"
    )

    if(SHADERC_LIB)
        target_compile_definitions(ace_be_vulkan PRIVATE SHADERC_AVAILABLE=1)
        target_link_libraries(ace_be_vulkan PRIVATE ${SHADERC_LIB})
        message(STATUS "shaderc found - Vulkan backend will support SPIR-V compilation")
    else()
        message(STATUS "shaderc not found - Vulkan backend will work but cannot compile shaders")
    endif()

    target_link_libraries(ace_be_vulkan PRIVATE
        Vulkan::Vulkan
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
