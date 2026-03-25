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

# GPU 后端测试
add_executable(gpu_test gpu_test.c)
target_include_directories(gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpu_test PRIVATE ace_core)
set_target_properties(gpu_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# CUDA 后端测试
add_executable(cuda_test cuda_test.c)
target_include_directories(cuda_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cuda_test PRIVATE ace_core)
set_target_properties(cuda_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Vulkan 后端测试
add_executable(vulkan_test vulkan_test.c)
target_include_directories(vulkan_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(vulkan_test PRIVATE ace_core)
set_target_properties(vulkan_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# OpenCL 后端测试
add_executable(opencl_test opencl_test.c)
target_include_directories(opencl_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(opencl_test PRIVATE ace_core)
set_target_properties(opencl_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 综合测试套件
add_executable(comprehensive_test comprehensive_test.c)
target_include_directories(comprehensive_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(comprehensive_test PRIVATE ace_core m)
set_target_properties(comprehensive_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Linux 下设置 RPATH
if(NOT WIN32)
    set_target_properties(ace_demo simple_test user_kernels gpu_test cuda_test vulkan_test opencl_test comprehensive_test PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/bin;${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN"
    )
endif()

install(TARGETS ace_demo simple_test user_kernels gpu_test cuda_test vulkan_test opencl_test comprehensive_test RUNTIME DESTINATION bin)

# ============================================================================
# CTest 测试配置
# ============================================================================

include(CTest)

# 基础测试
add_test(NAME SimpleTest COMMAND simple_test)
set_tests_properties(SimpleTest PROPERTIES
    LABELS "CPU"
    TIMEOUT 30
)

# GPU 后端测试
add_test(NAME CUDATest COMMAND cuda_test)
set_tests_properties(CUDATest PROPERTIES
    LABELS "GPU;CUDA"
    TIMEOUT 60
)

add_test(NAME OpenCLTest COMMAND opencl_test)
set_tests_properties(OpenCLTest PROPERTIES
    LABELS "GPU;OpenCL"
    TIMEOUT 60
)

add_test(NAME VulkanTest COMMAND vulkan_test)
set_tests_properties(VulkanTest PROPERTIES
    LABELS "GPU;Vulkan"
    TIMEOUT 60
)

# 综合测试
add_test(NAME ComprehensiveTest COMMAND comprehensive_test)
set_tests_properties(ComprehensiveTest PROPERTIES
    LABELS "All"
    TIMEOUT 180
)

# 多后端综合测试
add_test(NAME MultiGPUTest COMMAND gpu_test)
set_tests_properties(MultiGPUTest PROPERTIES
    LABELS "GPU;Multi"
    TIMEOUT 120
)

# 用户内核测试
add_test(NAME UserKernelsTest COMMAND user_kernels)
set_tests_properties(UserKernelsTest PROPERTIES
    LABELS "CPU"
    TIMEOUT 60
)

# Demo 测试
add_test(NAME DemoTest COMMAND ace_demo)
set_tests_properties(DemoTest PROPERTIES
    LABELS "All"
    TIMEOUT 120
)

# 测试分组
add_test(NAME AllCPUTests COMMAND ${CMAKE_COMMAND} -E echo "CPU Tests: simple_test, user_kernels")
add_test(NAME AllGPUTests COMMAND ${CMAKE_COMMAND} -E echo "GPU Tests: cuda_test, opencl_test, vulkan_test")
