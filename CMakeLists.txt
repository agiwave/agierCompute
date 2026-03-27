# =============================================================================
# AgierCompute - 跨平台 GPU 计算框架
# =============================================================================
cmake_minimum_required(VERSION 3.15)

# 项目名称和版本
project(AgierCompute 
    VERSION 1.0.0
    DESCRIPTION "Cross-platform GPU Computing Framework"
    LANGUAGES C CXX
)

# =============================================================================
# 全局配置
# =============================================================================

# C 标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# C++ 标准 (需要 C++20 支持 designated initializer)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 输出目录配置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 安装目录配置
include(GNUInstallDirs)
set(INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# 启用测试
enable_testing()

# =============================================================================
# 编译选项
# =============================================================================

# 导出符号
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# 位置无关代码
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 编译警告
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/W3)
endif()

# =============================================================================
# 子项目
# =============================================================================

# 核心库
add_subdirectory(src/core)

# 后端
add_subdirectory(src/cpu)
add_subdirectory(src/cuda)
add_subdirectory(src/opencl)
add_subdirectory(src/vulkan)

# 示例
add_subdirectory(examples)

# 测试
add_subdirectory(tests)

# =============================================================================
# 安装配置
# =============================================================================

# 安装头文件
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# =============================================================================
# CMake 配置文件导出
# =============================================================================

include(CMakePackageConfigHelpers)

# 生成版本配置文件
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# 生成配置文件
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${INSTALL_CMAKEDIR}
)

# 导出目标
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${INSTALL_CMAKEDIR}
)

# 安装配置文件
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${INSTALL_CMAKEDIR}
)

# =============================================================================
# 项目信息输出
# =============================================================================

message(STATUS "")
message(STATUS "=== ${PROJECT_NAME} Configuration ===")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C Compiler:     ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  C++ Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
