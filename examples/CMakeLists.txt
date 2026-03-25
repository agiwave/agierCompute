cmake_minimum_required(VERSION 3.15)

project(ace_examples VERSION 1.0.0 LANGUAGES C CXX)

# Simple test
add_executable(simple_test simple_test.c)
target_include_directories(simple_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(simple_test PRIVATE ace_core)

# 统一输出目录
set_target_properties(simple_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Demo (C++)
add_executable(ace_demo demo.cpp)

target_include_directories(ace_demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ace_demo PRIVATE ace_core)

# 统一输出目录
set_target_properties(ace_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 用户自定义内核示例
add_executable(user_kernels user_kernels.c)
target_include_directories(user_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(user_kernels PRIVATE ace_core)
set_target_properties(user_kernels PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Linux下设置RPATH，确保能找到库
if(NOT WIN32)
    set_target_properties(ace_demo simple_test user_kernels PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/bin;${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN"
    )
endif()

install(TARGETS ace_demo simple_test user_kernels RUNTIME DESTINATION bin)