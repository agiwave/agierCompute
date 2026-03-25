cmake_minimum_required(VERSION 3.15)

project(ace_examples VERSION 1.0.0 LANGUAGES C CXX)

# Simple test
add_executable(simple_test simple_test.c)
target_include_directories(simple_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(simple_test PRIVATE ace_core)
set_target_properties(simple_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Demo (C++)
add_executable(ace_demo demo.cpp)
target_include_directories(ace_demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(ace_demo PRIVATE ace_core)
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

# 多设备测试
add_executable(multi_device_test multi_device_test.c)
target_include_directories(multi_device_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(multi_device_test PRIVATE ace_core)
set_target_properties(multi_device_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 完整单元测试套件
add_executable(unit_tests unit_tests.c)
target_include_directories(unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(unit_tests PRIVATE ace_core m)
set_target_properties(unit_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 性能基准测试
add_executable(benchmark benchmark.c)
target_include_directories(benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(benchmark PRIVATE ace_core m)
set_target_properties(benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 神经网络示例
add_executable(neural_net neural_net.c)
target_include_directories(neural_net PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(neural_net PRIVATE ace_core m)
set_target_properties(neural_net PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Linux 下设置 RPATH，确保能找到库
if(NOT WIN32)
    set_target_properties(ace_demo simple_test user_kernels multi_device_test unit_tests benchmark neural_net PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/bin;${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN"
    )
endif()

install(TARGETS ace_demo simple_test user_kernels multi_device_test unit_tests benchmark neural_net RUNTIME DESTINATION bin)
