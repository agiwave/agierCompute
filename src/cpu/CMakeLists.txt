# =============================================================================
# AgierCompute CPU Backend (Placeholder)
# =============================================================================

# CPU 后端占位实现 - 不包含 JIT 编译
# 实际使用时请启用 GPU 后端 (CUDA/OpenCL/Vulkan)

add_library(ace_be_cpu SHARED
    cpu_backend.c
)

target_include_directories(ace_be_cpu
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_definitions(ace_be_cpu PRIVATE ACE_BACKEND_EXPORTS)

target_link_libraries(ace_be_cpu PRIVATE ace_core)

set_target_properties(ace_be_cpu PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME ace_be_cpu
    EXPORT_NAME ace_be_cpu
)

# 安装
install(TARGETS ace_be_cpu
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
