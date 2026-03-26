# =============================================================================
# AgierCompute Core Engine
# =============================================================================

add_library(ace_core SHARED
    engine_backend.c
    engine_device.c
    engine_memory.c
    engine_kernel.c
)

target_include_directories(ace_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_definitions(ace_core PRIVATE
    ACE_CORE_EXPORTS
)

# 链接系统库
if(UNIX)
    target_link_libraries(ace_core PRIVATE m)
endif()

set_target_properties(ace_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME ace_core
    EXPORT_NAME ace_core
)

# 安装
install(TARGETS ace_core
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
