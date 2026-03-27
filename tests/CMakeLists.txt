# =============================================================================
# AgierCompute Test Suite
# =============================================================================
#
# 测试架构:
#   test_backend.c      - 后端基础测试（设备属性、内存、标量参数）
#   test_api.c          - API 接口测试（设备枚举、错误处理、类型系统）
#   test_dtype.c        - 数据类型测试（9种类型 × 5种操作）
#   test_kernel.c       - 内核操作测试（用户自定义内核）
#   test_comprehensive.c - 综合测试（后端 × 类型 × 操作矩阵）
#   benchmark.c         - 性能基准测试
# =============================================================================

# =============================================================================
# 测试库
# =============================================================================

add_library(ace_test STATIC lib/ace_test.c)
target_include_directories(ace_test PRIVATE ${PROJECT_SOURCE_DIR}/include lib)

# =============================================================================
# 测试程序
# =============================================================================

# 1. 后端基础测试
add_executable(test_backend test_backend.c)
target_include_directories(test_backend PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_backend PRIVATE ace_core)

# 2. API 接口测试
add_executable(test_api test_api.c)
target_include_directories(test_api PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_api PRIVATE ace_core)

# 3. 数据类型测试
add_executable(test_dtype test_dtype.c)
target_include_directories(test_dtype PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(test_dtype PRIVATE ace_core)
else()
    target_link_libraries(test_dtype PRIVATE ace_core m)
endif()

# 4. 内核操作测试
add_executable(test_kernel test_kernel.c)
target_include_directories(test_kernel PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(test_kernel PRIVATE ace_core)
else()
    target_link_libraries(test_kernel PRIVATE ace_core m)
endif()

# 5. 综合测试
add_executable(test_comprehensive test_comprehensive.c)
target_include_directories(test_comprehensive PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(test_comprehensive PRIVATE ace_core)
else()
    target_link_libraries(test_comprehensive PRIVATE ace_core m)
endif()

# 6. 性能基准测试
add_executable(benchmark benchmark.c)
target_include_directories(benchmark PRIVATE ${PROJECT_SOURCE_DIR}/include lib)
if(WIN32)
    target_link_libraries(benchmark PRIVATE ace_core ace_test)
else()
    target_link_libraries(benchmark PRIVATE ace_core ace_test m)
endif()

# =============================================================================
# CTest 配置
# =============================================================================

enable_testing()

add_test(NAME Backend_Test COMMAND test_backend)
set_tests_properties(Backend_Test PROPERTIES LABELS "Core;Backend" TIMEOUT 60)

add_test(NAME API_Test COMMAND test_api)
set_tests_properties(API_Test PROPERTIES LABELS "Core;API" TIMEOUT 60)

add_test(NAME Data_Type_Test COMMAND test_dtype)
set_tests_properties(Data_Type_Test PROPERTIES LABELS "Core;DataType" TIMEOUT 300)

add_test(NAME Kernel_Test COMMAND test_kernel)
set_tests_properties(Kernel_Test PROPERTIES LABELS "Core;Kernel" TIMEOUT 300)

add_test(NAME Comprehensive_Test COMMAND test_comprehensive)
set_tests_properties(Comprehensive_Test PROPERTIES LABELS "Core;Comprehensive" TIMEOUT 600)

# =============================================================================
# 自定义目标
# =============================================================================

add_custom_target(test-quick
    COMMAND ${CMAKE_CTEST_COMMAND} -L "Core" --output-on-failure
    DEPENDS test_backend test_api test_dtype test_kernel
    COMMENT "Running core tests..."
)

add_custom_target(test-all
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_backend test_api test_dtype test_kernel test_comprehensive
    COMMENT "Running all tests..."
)

add_custom_target(test-benchmark
    COMMAND benchmark
    DEPENDS benchmark
    COMMENT "Running benchmarks..."
)
