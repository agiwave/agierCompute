cmake_minimum_required(VERSION 3.15)

project(ace_core VERSION 1.0.0 LANGUAGES C)

add_library(ace_core SHARED engine.c)

target_include_directories(ace_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(ace_core PRIVATE ACE_CORE_EXPORTS)

# Linux 下需要 dl 库用于动态加载
if(NOT WIN32)
    target_link_libraries(ace_core PRIVATE dl)
endif()

# 统一输出目录
set_target_properties(ace_core PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(MSVC)
    target_compile_options(ace_core PRIVATE /W3)
else()
    target_compile_options(ace_core PRIVATE -Wall -Wextra)
endif()

# Install
install(TARGETS ace_core LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
