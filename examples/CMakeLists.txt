# =============================================================================
# AgierCompute Examples and Tests
# =============================================================================

# 1. 基础测试 - CPU 后端
add_executable(simple_test simple_test.c)
target_include_directories(simple_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(simple_test PRIVATE ace_core)

# 2. CUDA 后端测试
add_executable(test_cuda cuda_test.c)
target_include_directories(test_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_cuda PRIVATE ace_core)

# 3. OpenCL 后端测试
add_executable(test_opencl opencl_test.c)
target_include_directories(test_opencl PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_opencl PRIVATE ace_core)

# 4. Vulkan 后端测试
add_executable(test_vulkan vulkan_test.c)
target_include_directories(test_vulkan PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_vulkan PRIVATE ace_core)

# 5. 用户内核测试
add_executable(test_kernels kernel_tests.c)
target_include_directories(test_kernels PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(test_kernels PRIVATE ace_core)
else()
    target_link_libraries(test_kernels PRIVATE ace_core m)
endif()

# 6. 数据类型测试（包含 AI 数据类型：FLOAT16, BFLOAT16, INT8, UINT8, INT16）
add_executable(test_dtypes test_dtypes.c)
target_include_directories(test_dtypes PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(test_dtypes PRIVATE ace_core)
else()
    target_link_libraries(test_dtypes PRIVATE ace_core m)
endif()

# 6b. 数据类型调试测试
add_executable(test_dtype_debug test_dtype_debug.c)
target_include_directories(test_dtype_debug PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(test_dtype_debug PRIVATE ace_core)
else()
    target_link_libraries(test_dtype_debug PRIVATE ace_core m)
endif()

# 7. 用户自定义内核示例
add_executable(example_user_kernels user_kernels.c)
target_include_directories(example_user_kernels PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(example_user_kernels PRIVATE ace_core)

# 8. C++ Demo
add_executable(demo demo.cpp)
target_include_directories(demo PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(demo PRIVATE ace_core)

# 9. 设备遍历示例
add_executable(example_device_enum device_enum.c)
target_include_directories(example_device_enum PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(example_device_enum PRIVATE ace_core)

# 10. 完整数据类型测试 - 覆盖所有后端和所有数据类型
add_executable(test_all_dtypes test_all_dtypes.c)
target_include_directories(test_all_dtypes PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(test_all_dtypes PRIVATE ace_core)
else()
    target_link_libraries(test_all_dtypes PRIVATE ace_core m)
endif()

# 11. 性能基准测试
add_library(ace_test STATIC lib/ace_test.c)
target_include_directories(ace_test PRIVATE ${PROJECT_SOURCE_DIR}/include lib)

add_executable(benchmark benchmark.c)
target_include_directories(benchmark PRIVATE ${PROJECT_SOURCE_DIR}/include lib)
target_link_libraries(benchmark PRIVATE ace_core ace_test m)

# 12. 三维综合测试 - 多后端/多数据类型/多数学函数
add_executable(comprehensive_test comprehensive_test.c)
target_include_directories(comprehensive_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(comprehensive_test PRIVATE ace_core)
else()
    target_link_libraries(comprehensive_test PRIVATE ace_core m)
endif()

# =============================================================================
# CTest 测试配置
# =============================================================================

enable_testing()

# 后端测试
add_test(NAME CUDA_Backend COMMAND test_cuda)
set_tests_properties(CUDA_Backend PROPERTIES
    LABELS "CUDA;Backend"
    TIMEOUT 60
)

add_test(NAME OpenCL_Backend COMMAND test_opencl)
set_tests_properties(OpenCL_Backend PROPERTIES
    LABELS "OpenCL;Backend"
    TIMEOUT 60
)

add_test(NAME Vulkan_Backend COMMAND test_vulkan)
set_tests_properties(Vulkan_Backend PROPERTIES
    LABELS "Vulkan;Backend"
    TIMEOUT 60
)

# 内核功能测试
add_test(NAME All_Kernels COMMAND test_kernels)
set_tests_properties(All_Kernels PROPERTIES
    LABELS "All;Kernels;Functionality"
    TIMEOUT 600
)

# 数据类型测试 - 完整测试所有数据类型 (包括 FP16/BF16)
add_test(NAME Data_Types COMMAND test_dtypes)
set_tests_properties(Data_Types PROPERTIES
    LABELS "All;DataTypes;Functionality;AI"
    TIMEOUT 300
)

# 完整数据类型测试 - 所有后端统一测试
add_test(NAME All_DataTypes COMMAND test_all_dtypes)
set_tests_properties(All_DataTypes PROPERTIES
    LABELS "All;DataTypes;CUDA;OpenCL;Vulkan;FP16;BF16"
    TIMEOUT 300
)

# 三维综合测试
add_test(NAME Comprehensive_Test COMMAND comprehensive_test)
set_tests_properties(Comprehensive_Test PROPERTIES
    LABELS "All;Comprehensive;3D-Test"
    TIMEOUT 600
)

# 示例程序测试
add_test(NAME Example_UserKernels COMMAND example_user_kernels)
set_tests_properties(Example_UserKernels PROPERTIES
    LABELS "Example;UserKernels"
    TIMEOUT 60
)

add_test(NAME Demo_CPP COMMAND demo)
set_tests_properties(Demo_CPP PROPERTIES
    LABELS "Example;Demo;CPP"
    TIMEOUT 120
)

# =============================================================================
# 自定义测试目标
# =============================================================================

add_custom_target(test-quick
    COMMAND ${CMAKE_CTEST_COMMAND} -L "Quick" --output-on-failure
    DEPENDS simple_test
    COMMENT "Running quick tests..."
)

add_custom_target(test-backends
    COMMAND ${CMAKE_CTEST_COMMAND} -L "Backend" --output-on-failure
    DEPENDS simple_test test_cuda test_opencl test_vulkan
    COMMENT "Running backend tests..."
)

add_custom_target(test-functionality
    COMMAND ${CMAKE_CTEST_COMMAND} -L "Functionality" --output-on-failure
    DEPENDS test_kernels test_dtypes
    COMMENT "Running functionality tests..."
)

add_custom_target(test-all
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS simple_test test_cuda test_opencl test_vulkan test_kernels test_dtypes example_user_kernels demo
    COMMENT "Running all tests..."
)

add_custom_target(test-cuda
    COMMAND ${CMAKE_CTEST_COMMAND} -L "CUDA" --output-on-failure
    DEPENDS test_cuda
    COMMENT "Running CUDA tests..."
)

add_custom_target(test-opencl
    COMMAND ${CMAKE_CTEST_COMMAND} -L "OpenCL" --output-on-failure
    DEPENDS test_opencl
    COMMENT "Running OpenCL tests..."
)

add_custom_target(test-vulkan
    COMMAND ${CMAKE_CTEST_COMMAND} -L "Vulkan" --output-on-failure
    DEPENDS test_vulkan
    COMMENT "Running Vulkan tests..."
)

# =============================================================================
# 安装规则（只安装库和头文件，不安装测试程序）
# =============================================================================

# 测试程序不安装，只保留在 build/bin 目录
