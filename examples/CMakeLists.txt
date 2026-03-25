cmake_minimum_required(VERSION 3.15)

project(ace_examples VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

# ============================================================================
# 测试程序
# ============================================================================

# 1. 基础测试 - CPU 后端
add_executable(simple_test simple_test.c)
target_include_directories(simple_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(simple_test PRIVATE ace_core)
set_target_properties(simple_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 2. CUDA 后端测试
add_executable(test_cuda cuda_test.c)
target_include_directories(test_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_cuda PRIVATE ace_core)
set_target_properties(test_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 3. OpenCL 后端测试
add_executable(test_opencl opencl_test.c)
target_include_directories(test_opencl PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_opencl PRIVATE ace_core)
set_target_properties(test_opencl PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 4. Vulkan 后端测试
add_executable(test_vulkan vulkan_test.c)
target_include_directories(test_vulkan PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_vulkan PRIVATE ace_core)
set_target_properties(test_vulkan PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 5. 用户内核测试 - 测试所有用户定义的内核
add_executable(test_kernels kernel_tests.c)
target_include_directories(test_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_kernels PRIVATE ace_core m)
set_target_properties(test_kernels PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 6. 用户自定义内核示例
add_executable(example_user_kernels user_kernels.c)
target_include_directories(example_user_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(example_user_kernels PRIVATE ace_core)
set_target_properties(example_user_kernels PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# 7. C++ Demo
add_executable(demo demo.cpp)
target_include_directories(demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(demo PRIVATE ace_core)
set_target_properties(demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ============================================================================
# RPATH 设置
# ============================================================================

if(NOT WIN32)
    set_target_properties(simple_test test_cuda test_opencl test_vulkan test_kernels 
                          example_user_kernels demo PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/bin;${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN"
    )
endif()

# ============================================================================
# CTest 测试配置
# ============================================================================

enable_testing()

# 快速测试（1 分钟内）
add_test(NAME QuickTest COMMAND simple_test)
set_tests_properties(QuickTest PROPERTIES
    LABELS "CPU;Quick"
    TIMEOUT 60
)

# 后端测试
add_test(NAME CUDABackend COMMAND test_cuda)
set_tests_properties(CUDABackend PROPERTIES
    LABELS "CUDA;Backend"
    TIMEOUT 60
)

add_test(NAME OpenCLBackend COMMAND test_opencl)
set_tests_properties(OpenCLBackend PROPERTIES
    LABELS "OpenCL;Backend"
    TIMEOUT 60
)

add_test(NAME VulkanBackend COMMAND test_vulkan)
set_tests_properties(VulkanBackend PROPERTIES
    LABELS "Vulkan;Backend"
    TIMEOUT 60
)

# 全面内核测试
add_test(NAME AllKernels COMMAND test_kernels)
set_tests_properties(AllKernels PROPERTIES
    LABELS "All;Kernels"
    TIMEOUT 600
)

# ============================================================================
# 自定义测试目标
# ============================================================================

# 运行所有快速测试
add_custom_target(test-quick
    COMMAND ${CMAKE_CTEST_COMMAND} -L "Quick" --output-on-failure
    DEPENDS simple_test
    COMMENT "Running quick tests..."
)

# 运行所有后端测试
add_custom_target(test-backends
    COMMAND ${CMAKE_CTEST_COMMAND} -L "Backend" --output-on-failure
    DEPENDS test_cuda test_opencl test_vulkan
    COMMENT "Running backend tests..."
)

# 运行所有内核测试
add_custom_target(test-all
    COMMAND ${CMAKE_CTEST_COMMAND} -L "All" --output-on-failure
    DEPENDS test_kernels
    COMMENT "Running all kernel tests..."
)

# ============================================================================
# 安装规则
# ============================================================================

install(TARGETS 
    simple_test test_cuda test_opencl test_vulkan test_kernels
    example_user_kernels demo
    RUNTIME DESTINATION bin
)
