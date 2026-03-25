# =============================================================================
# AgierCompute Core Library
# =============================================================================

add_library(ace_core SHARED
    engine.c
)

target_include_directories(ace_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(ace_core PRIVATE ACE_CORE_EXPORTS)

# 链接系统库
if(UNIX)
    target_link_libraries(ace_core PRIVATE dl m)
endif()

# 设置属性
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
