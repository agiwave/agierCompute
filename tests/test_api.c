/**
 * @file test_api.c
 * @brief API 接口测试 - 测试框架基础 API 功能
 *
 * 测试内容：
 * - 设备枚举和属性
 * - 错误处理
 * - 内存管理基础
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ace.h"

/* 测试统计 */
static int g_passed = 0;
static int g_failed = 0;

#define TEST(name, cond) do { \
    printf("  %-30s ", name); \
    if (cond) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while(0)

/* ============================================================================
 * 设备 API 测试
 * ============================================================================ */

static void test_device_api(void) {
    printf("\n--- Device API Tests ---\n");
    
    /* 测试设备计数 */
    int cuda_count = 0, opencl_count = 0, vulkan_count = 0, total = 0;
    
    ace_error_t err = ace_device_count(ACE_DEVICE_CUDA, &cuda_count);
    TEST("ace_device_count(CUDA)", err == ACE_OK);
    
    err = ace_device_count(ACE_DEVICE_OPENCL, &opencl_count);
    TEST("ace_device_count(OpenCL)", err == ACE_OK);
    
    err = ace_device_count(ACE_DEVICE_VULKAN, &vulkan_count);
    TEST("ace_device_count(Vulkan)", err == ACE_OK);
    
    err = ace_device_count(ACE_DEVICE_ALL, &total);
    TEST("ace_device_count(ALL)", err == ACE_OK && total >= cuda_count + opencl_count + vulkan_count);
    
    /* 测试设备属性 */
    if (cuda_count > 0) {
        ace_device_t dev = NULL;
        err = ace_device_get(ACE_DEVICE_CUDA, 0, &dev);
        TEST("ace_device_get(CUDA, 0)", err == ACE_OK && dev != NULL);
        
        if (dev) {
            ace_device_props_t props;
            err = ace_device_props(dev, &props);
            TEST("ace_device_props()", err == ACE_OK);
            TEST("props.name valid", props.name[0] != '\0');
            TEST("props.memory > 0", props.total_memory > 0);
            TEST("props.type == CUDA", props.type == ACE_DEVICE_CUDA);
            
            ace_device_release(dev);
        }
    }
    
    /* 测试无效设备索引 */
    ace_device_t invalid_dev = NULL;
    err = ace_device_get(ACE_DEVICE_CUDA, 999, &invalid_dev);
    TEST("ace_device_get(invalid index)", err != ACE_OK || invalid_dev == NULL);
}

/* ============================================================================
 * 内存 API 测试
 * ============================================================================ */

static void test_memory_api(void) {
    printf("\n--- Memory API Tests ---\n");
    
    /* 找一个可用设备 */
    ace_device_t dev = NULL;
    int count = 0;
    
    if (ace_device_count(ACE_DEVICE_CUDA, &count) == ACE_OK && count > 0) {
        ace_device_get(ACE_DEVICE_CUDA, 0, &dev);
    } else if (ace_device_count(ACE_DEVICE_OPENCL, &count) == ACE_OK && count > 0) {
        ace_device_get(ACE_DEVICE_OPENCL, 0, &dev);
    } else if (ace_device_count(ACE_DEVICE_VULKAN, &count) == ACE_OK && count > 0) {
        ace_device_get(ACE_DEVICE_VULKAN, 0, &dev);
    }
    
    if (!dev) {
        printf("  (No device available, skipping memory tests)\n");
        return;
    }
    
    /* 测试内存分配 */
    ace_buffer_t buf = NULL;
    ace_error_t err = ace_buffer_alloc(dev, 1024, &buf);
    TEST("ace_buffer_alloc(1024)", err == ACE_OK && buf != NULL);
    
    if (buf) {
        /* 测试写入 */
        char data[] = "Hello, AgierCompute!";
        err = ace_buffer_write(buf, data, sizeof(data));
        TEST("ace_buffer_write()", err == ACE_OK);
        
        /* 测试读取 */
        char read_back[64] = {0};
        err = ace_buffer_read(buf, read_back, sizeof(data));
        TEST("ace_buffer_read()", err == ACE_OK);
        TEST("read back correct", strcmp(read_back, data) == 0);
        
        /* 测试释放 */
        ace_buffer_free(buf);
        TEST("ace_buffer_free()", 1); /* 只要没崩溃就算通过 */
    }
    
    /* 测试大内存分配 */
    ace_buffer_t big_buf = NULL;
    err = ace_buffer_alloc(dev, 16 * 1024 * 1024, &big_buf); /* 16 MB */
    TEST("ace_buffer_alloc(16MB)", err == ACE_OK && big_buf != NULL);
    if (big_buf) ace_buffer_free(big_buf);
    
    ace_device_release(dev);
}

/* ============================================================================
 * 错误处理测试
 * ============================================================================ */

static void test_error_handling(void) {
    printf("\n--- Error Handling Tests ---\n");
    
    /* 测试错误字符串 */
    const char* str = ace_error_string(ACE_OK);
    TEST("ace_error_string(ACE_OK)", str != NULL && strlen(str) > 0);
    
    str = ace_error_string(ACE_ERROR);
    TEST("ace_error_string(ACE_ERROR)", str != NULL && strlen(str) > 0);
    
    str = ace_error_string(ACE_ERROR_MEM);
    TEST("ace_error_string(ACE_ERROR_MEM)", str != NULL && strlen(str) > 0);
    
    /* 测试无效参数 */
    int count = 0;
    ace_error_t err = ace_device_count((ace_device_type_t)999, &count);
    TEST("ace_device_count(invalid type)", err == ACE_OK); /* 应该返回 0 设备 */
}

/* ============================================================================
 * 数据类型 API 测试
 * ============================================================================ */

static void test_dtype_api(void) {
    printf("\n--- Data Type API Tests ---\n");
    
    /* 测试类型大小 */
    TEST("ace_dtype_size(FLOAT32)", ace_dtype_size(ACE_DTYPE_FLOAT32) == 4);
    TEST("ace_dtype_size(FLOAT64)", ace_dtype_size(ACE_DTYPE_FLOAT64) == 8);
    TEST("ace_dtype_size(INT32)", ace_dtype_size(ACE_DTYPE_INT32) == 4);
    TEST("ace_dtype_size(INT64)", ace_dtype_size(ACE_DTYPE_INT64) == 8);
    TEST("ace_dtype_size(FLOAT16)", ace_dtype_size(ACE_DTYPE_FLOAT16) == 2);
    TEST("ace_dtype_size(BFLOAT16)", ace_dtype_size(ACE_DTYPE_BFLOAT16) == 2);
    TEST("ace_dtype_size(INT8)", ace_dtype_size(ACE_DTYPE_INT8) == 1);
    TEST("ace_dtype_size(UINT8)", ace_dtype_size(ACE_DTYPE_UINT8) == 1);
    TEST("ace_dtype_size(INT16)", ace_dtype_size(ACE_DTYPE_INT16) == 2);
    
    /* 测试类型名称 */
    TEST("ace_dtype_name(FLOAT32)", strcmp(ace_dtype_name(ACE_DTYPE_FLOAT32), "float32") == 0);
    TEST("ace_dtype_name(FLOAT64)", strcmp(ace_dtype_name(ACE_DTYPE_FLOAT64), "float64") == 0);
    TEST("ace_dtype_name(INT32)", strcmp(ace_dtype_name(ACE_DTYPE_INT32), "int32") == 0);
}

/* ============================================================================
 * 浮点转换测试
 * ============================================================================ */

static void test_fp_conversion(void) {
    printf("\n--- FP Conversion Tests ---\n");
    
    /* FP16 转换 */
    ace_float16_t h = float_to_float16(1.0f);
    float f = float16_to_float(h);
    TEST("FP16 roundtrip(1.0)", f > 0.99f && f < 1.01f);
    
    h = float_to_float16(0.5f);
    f = float16_to_float(h);
    TEST("FP16 roundtrip(0.5)", f > 0.49f && f < 0.51f);
    
    h = float_to_float16(-1.0f);
    f = float16_to_float(h);
    TEST("FP16 roundtrip(-1.0)", f < -0.99f && f > -1.01f);
    
    /* BF16 转换 */
    ace_bfloat16_t bf = float_to_bfloat16(1.0f);
    f = bfloat16_to_float(bf);
    TEST("BF16 roundtrip(1.0)", f > 0.99f && f < 1.01f);
    
    bf = float_to_bfloat16(0.5f);
    f = bfloat16_to_float(bf);
    TEST("BF16 roundtrip(0.5)", f > 0.49f && f < 0.51f);
    
    bf = float_to_bfloat16(-1.0f);
    f = bfloat16_to_float(bf);
    TEST("BF16 roundtrip(-1.0)", f < -0.99f && f > -1.01f);
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main() {
    printf("========================================================\n");
    printf("  AgierCompute - API Test Suite\n");
    printf("========================================================\n");
    
    test_device_api();
    test_memory_api();
    test_error_handling();
    test_dtype_api();
    test_fp_conversion();
    
    printf("\n========================================================\n");
    printf("  Results: %d passed, %d failed\n", g_passed, g_failed);
    printf("========================================================\n");
    
    return (g_failed > 0) ? 1 : 0;
}
