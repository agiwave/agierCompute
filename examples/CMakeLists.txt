# =============================================================================
# AgierCompute Examples
# =============================================================================
#
# 示例程序:
#   example_device_enum.c  - 设备枚举示例
#   example_user_kernels.c - 用户自定义内核示例
#   demo.cpp               - C++ 示例
# =============================================================================

# 1. 设备枚举示例
add_executable(example_device_enum example_device_enum.c)
target_include_directories(example_device_enum PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(example_device_enum PRIVATE ace_core)

# 2. 用户自定义内核示例
add_executable(example_user_kernels example_user_kernels.c)
target_include_directories(example_user_kernels PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(example_user_kernels PRIVATE ace_core)
else()
    target_link_libraries(example_user_kernels PRIVATE ace_core m)
endif()

# 3. C++ Demo
add_executable(demo demo.cpp)
target_include_directories(demo PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(demo PRIVATE ace_core)

# =============================================================================
# CTest 配置（示例程序测试）
# =============================================================================

add_test(NAME Example_DeviceEnum COMMAND example_device_enum)
set_tests_properties(Example_DeviceEnum PROPERTIES LABELS "Example" TIMEOUT 60)

add_test(NAME Example_UserKernels COMMAND example_user_kernels)
set_tests_properties(Example_UserKernels PROPERTIES LABELS "Example" TIMEOUT 60)

add_test(NAME Demo_CPP COMMAND demo)
set_tests_properties(Demo_CPP PROPERTIES LABELS "Example" TIMEOUT 120)