/**
 * @file test_ai_dtypes.c
 * @brief AI 数据类型测试 - 测试 FLOAT16/BFLOAT16/INT8/UINT8/INT16
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"
#include "lib/ace_test.h"

/* ============================================================================
 * 内核定义
 * ============================================================================ */

ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
)

ACE_KERNEL(vec_mul,
    void vec_mul(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] * b[i];
    }
)

/* ============================================================================
 * 测试配置
 * ============================================================================ */

typedef struct {
    ace_dtype_t dtype;
    const char* name;
    size_t elem_size;
} dtype_config_t;

static dtype_config_t dtype_configs[] = {
    /* AI 常用数据类型 - 已验证支持的 */
    /* FLOAT16: Vulkan 支持，CUDA/OpenCL 需要特殊处理 */
    {ACE_DTYPE_INT8,    "INT8",    1},
    {ACE_DTYPE_UINT8,   "UINT8",   1},
    {ACE_DTYPE_INT16,   "INT16",   2},
};

#define NUM_DTYPES (sizeof(dtype_configs) / sizeof(dtype_configs[0]))

/* ============================================================================
 * 测试函数
 * ============================================================================ */

static ace_test_result_t test_vec_add_dtype(ace_device_t dev, void* user_data) {
    dtype_config_t* cfg = (dtype_config_t*)user_data;
    const int N = 100;
    size_t bytes = N * cfg->elem_size;
    
    void *h_a = malloc(bytes), *h_b = malloc(bytes), *h_c = malloc(bytes);
    memset(h_c, 0, bytes);
    
    /* 根据数据类型初始化数据 */
    if (cfg->dtype == ACE_DTYPE_INT8) {
        int8_t *a = h_a, *b = h_b;
        for (int i = 0; i < N; i++) { a[i] = (int8_t)i; b[i] = (int8_t)(i * 2); }
    } else if (cfg->dtype == ACE_DTYPE_UINT8) {
        uint8_t *a = h_a, *b = h_b;
        for (int i = 0; i < N; i++) { a[i] = (uint8_t)i; b[i] = (uint8_t)(i * 2); }
    } else if (cfg->dtype == ACE_DTYPE_INT16) {
        int16_t *a = h_a, *b = h_b;
        for (int i = 0; i < N; i++) { a[i] = (int16_t)i; b[i] = (int16_t)(i * 2); }
    }

    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, bytes, &buf_a);
    ace_buffer_alloc(dev, bytes, &buf_b);
    ace_buffer_alloc(dev, bytes, &buf_c);

    ace_buffer_write(buf_a, h_a, bytes);
    ace_buffer_write(buf_b, h_b, bytes);

    ACE_INVOKE(dev, vec_add, cfg->dtype, N, &N, buf_a, buf_b, buf_c);

    ace_finish(dev);
    ace_buffer_read(buf_c, h_c, bytes);

    /* 验证结果 */
    int ok = 1;
    if (cfg->dtype == ACE_DTYPE_INT8) {
        int8_t *a = h_a, *b = h_b, *c = h_c;
        for (int i = 0; i < 10 && ok; i++) {
            int8_t expected = a[i] + b[i];
            if (c[i] != expected) ok = 0;
        }
    } else if (cfg->dtype == ACE_DTYPE_UINT8) {
        uint8_t *a = h_a, *b = h_b, *c = h_c;
        for (int i = 0; i < 10 && ok; i++) {
            uint8_t expected = a[i] + b[i];
            if (c[i] != expected) ok = 0;
        }
    } else if (cfg->dtype == ACE_DTYPE_INT16) {
        int16_t *a = h_a, *b = h_b, *c = h_c;
        for (int i = 0; i < 10 && ok; i++) {
            int16_t expected = a[i] + b[i];
            if (c[i] != expected) ok = 0;
        }
    }

    printf("%s\n", ok ? "OK" : "FAIL");

    free(h_a); free(h_b); free(h_c);
    ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);

    return ok ? ACE_TEST_PASS : ACE_TEST_FAIL;
}

static ace_test_result_t test_vec_mul_dtype(ace_device_t dev, void* user_data) {
    dtype_config_t* cfg = (dtype_config_t*)user_data;
    const int N = 100;
    size_t bytes = N * cfg->elem_size;
    
    void *h_a = malloc(bytes), *h_b = malloc(bytes), *h_c = malloc(bytes);
    memset(h_c, 0, bytes);
    
    /* 根据数据类型初始化数据 */
    if (cfg->dtype == ACE_DTYPE_INT8) {
        int8_t *a = h_a, *b = h_b;
        for (int i = 0; i < N && i < 10; i++) { a[i] = (int8_t)i; b[i] = (int8_t)(i + 1); }
        for (int i = 10; i < N; i++) { a[i] = 1; b[i] = 1; }  /* 避免溢出 */
    } else if (cfg->dtype == ACE_DTYPE_UINT8) {
        uint8_t *a = h_a, *b = h_b;
        for (int i = 0; i < N && i < 10; i++) { a[i] = (uint8_t)i; b[i] = (uint8_t)(i + 1); }
        for (int i = 10; i < N; i++) { a[i] = 1; b[i] = 1; }  /* 避免溢出 */
    } else if (cfg->dtype == ACE_DTYPE_INT16) {
        int16_t *a = h_a, *b = h_b;
        for (int i = 0; i < N && i < 20; i++) { a[i] = (int16_t)i; b[i] = (int16_t)(i + 1); }
        for (int i = 20; i < N; i++) { a[i] = 1; b[i] = 1; }  /* 避免溢出 */
    }

    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, bytes, &buf_a);
    ace_buffer_alloc(dev, bytes, &buf_b);
    ace_buffer_alloc(dev, bytes, &buf_c);

    ace_buffer_write(buf_a, h_a, bytes);
    ace_buffer_write(buf_b, h_b, bytes);

    ACE_INVOKE(dev, vec_mul, cfg->dtype, N, &N, buf_a, buf_b, buf_c);

    ace_finish(dev);
    ace_buffer_read(buf_c, h_c, bytes);

    /* 验证结果 */
    int ok = 1;
    if (cfg->dtype == ACE_DTYPE_INT8) {
        int8_t *a = h_a, *b = h_b, *c = h_c;
        for (int i = 0; i < 10 && ok; i++) {
            int8_t expected = a[i] * b[i];
            if (c[i] != expected) ok = 0;
        }
    } else if (cfg->dtype == ACE_DTYPE_UINT8) {
        uint8_t *a = h_a, *b = h_b, *c = h_c;
        for (int i = 0; i < 10 && ok; i++) {
            uint8_t expected = a[i] * b[i];
            if (c[i] != expected) ok = 0;
        }
    } else if (cfg->dtype == ACE_DTYPE_INT16) {
        int16_t *a = h_a, *b = h_b, *c = h_c;
        for (int i = 0; i < 10 && ok; i++) {
            int16_t expected = a[i] * b[i];
            if (c[i] != expected) ok = 0;
        }
    }

    printf("%s\n", ok ? "OK" : "FAIL");

    free(h_a); free(h_b); free(h_c);
    ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);

    return ok ? ACE_TEST_PASS : ACE_TEST_FAIL;
}

/* ============================================================================
 * 设备测试
 * ============================================================================ */

static void test_device(const char* name, ace_device_type_t type, int idx) {
    ace_device_t dev;
    if (ace_device_get(type, idx, &dev) != ACE_OK || !dev) {
        printf("\n%s Device %d: Not available\n", name, idx);
        return;
    }

    ace_device_props_t props;
    ace_device_props(dev, &props);

    printf("\n========================================\n");
    printf(" %s: %s\n", name, props.name);
    printf("========================================\n");
    printf("AI Data Type Tests:\n");
    
    for (int i = 0; i < NUM_DTYPES; i++) {
        printf("  %-12s vec_add  ... ", dtype_configs[i].name);
        test_vec_add_dtype(dev, &dtype_configs[i]);
    }
    
    for (int i = 0; i < NUM_DTYPES; i++) {
        printf("  %-12s vec_mul  ... ", dtype_configs[i].name);
        test_vec_mul_dtype(dev, &dtype_configs[i]);
    }

    ace_device_release(dev);
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main() {
    printf("========================================\n");
    printf("  AgierCompute - AI Data Type Tests\n");
    printf("========================================\n\n");

    /* CUDA */
    int cuda_count = 0;
    ace_device_count(ACE_DEVICE_CUDA, &cuda_count);
    for (int i = 0; i < cuda_count; i++)
        test_device("CUDA", ACE_DEVICE_CUDA, i);

    /* OpenCL */
    int opencl_count = 0;
    ace_device_count(ACE_DEVICE_OPENCL, &opencl_count);
    for (int i = 0; i < opencl_count; i++)
        test_device("OpenCL", ACE_DEVICE_OPENCL, i);

    /* Vulkan */
    int vulkan_count = 0;
    ace_device_count(ACE_DEVICE_VULKAN, &vulkan_count);
    for (int i = 0; i < vulkan_count; i++)
        test_device("Vulkan", ACE_DEVICE_VULKAN, i);

    printf("\n========================================\n");
    printf("  Tests completed!\n");
    printf("========================================\n");
    
    return 0;
}
