/**
 * @file test_dtypes.c
 * @brief 测试 AI 重要数据类型 (FLOAT16/BFLOAT16/INT8 等)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"

/* ============================================================================
 * 内核定义 - 支持多种数据类型
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

ACE_KERNEL(relu,
    void relu(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] > 0 ? in[i] : 0;
    }
)

/* ============================================================================
 * 测试辅助
 * ============================================================================ */

static int g_passed = 0;
static int g_total = 0;

#define ASSERT_NEAR(a, b, eps) do { \
    if (fabs((double)(a) - (double)(b)) > (eps)) { \
        printf("FAIL: %g != %g\n", (double)(a), (double)(b)); \
        return 0; \
    } \
} while(0)

#define RUN_TEST(fn, dev, dtype, name) do { \
    g_total++; \
    printf("  %-20s ... ", name); \
    if (fn(dev, dtype)) { g_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

/* ============================================================================
 * FLOAT32 测试 (基准)
 * ============================================================================ */

static int test_vec_add_float32(ace_device_t dev, ace_dtype_t dtype) {
    (void)dtype;
    const int N = 20;
    float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) { a[i] = i * 1.0f; b[i] = i * 2.0f; }
    
    ace_buffer_t ba, bb, bc;
    ace_buffer_alloc(dev, N * sizeof(float), &ba);
    ace_buffer_alloc(dev, N * sizeof(float), &bb);
    ace_buffer_alloc(dev, N * sizeof(float), &bc);
    
    ace_buffer_write(ba, a, N * sizeof(float));
    ace_buffer_write(bb, b, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, ba, bb, bc};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);
    ace_buffer_read(bc, c, N * sizeof(float));
    
    for (int i = 0; i < 10; i++) ASSERT_NEAR(c[i], a[i] + b[i], 1e-5f);
    
    ace_buffer_free(ba); ace_buffer_free(bb); ace_buffer_free(bc);
    return 1;
}

/* ============================================================================
 * FLOAT16 测试
 * ============================================================================ */

static int test_vec_add_float16(ace_device_t dev, ace_dtype_t dtype) {
    (void)dtype;
    const int N = 20;
    float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) { a[i] = i * 0.5f; b[i] = i * 0.25f; }
    
    /* 转换为 FLOAT16 */
    ace_float16_t a16[N], b16[N];
    for (int i = 0; i < N; i++) {
        a16[i] = float_to_float16(a[i]);
        b16[i] = float_to_float16(b[i]);
    }
    
    ace_buffer_t ba, bb, bc;
    ace_buffer_alloc(dev, N * sizeof(ace_float16_t), &ba);
    ace_buffer_alloc(dev, N * sizeof(ace_float16_t), &bb);
    ace_buffer_alloc(dev, N * sizeof(ace_float16_t), &bc);
    
    ace_buffer_write(ba, a16, N * sizeof(ace_float16_t));
    ace_buffer_write(bb, b16, N * sizeof(ace_float16_t));
    
    int n = N;
    void* args[] = {&n, ba, bb, bc};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT16, N, args, types, 4);
    ace_finish(dev);
    
    /* 读取并转换回 float */
    ace_float16_t c16[N];
    ace_buffer_read(bc, c16, N * sizeof(ace_float16_t));
    for (int i = 0; i < N; i++) c[i] = float16_to_float(c16[i]);
    
    /* FLOAT16 精度较低，使用较大容差 */
    for (int i = 0; i < 10; i++) ASSERT_NEAR(c[i], a[i] + b[i], 0.01f);
    
    ace_buffer_free(ba); ace_buffer_free(bb); ace_buffer_free(bc);
    return 1;
}

/* ============================================================================
 * BFLOAT16 测试
 * ============================================================================ */

static int test_vec_add_bfloat16(ace_device_t dev, ace_dtype_t dtype) {
    (void)dtype;
    const int N = 20;
    float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) { a[i] = i * 0.5f; b[i] = i * 0.25f; }
    
    /* 转换为 BFLOAT16 */
    ace_bfloat16_t a16[N], b16[N];
    for (int i = 0; i < N; i++) {
        a16[i] = float_to_bfloat16(a[i]);
        b16[i] = float_to_bfloat16(b[i]);
    }
    
    ace_buffer_t ba, bb, bc;
    ace_buffer_alloc(dev, N * sizeof(ace_bfloat16_t), &ba);
    ace_buffer_alloc(dev, N * sizeof(ace_bfloat16_t), &bb);
    ace_buffer_alloc(dev, N * sizeof(ace_bfloat16_t), &bc);
    
    ace_buffer_write(ba, a16, N * sizeof(ace_bfloat16_t));
    ace_buffer_write(bb, b16, N * sizeof(ace_bfloat16_t));
    
    int n = N;
    void* args[] = {&n, ba, bb, bc};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_BFLOAT16, N, args, types, 4);
    ace_finish(dev);
    
    /* 读取并转换回 float */
    ace_bfloat16_t c16[N];
    ace_buffer_read(bc, c16, N * sizeof(ace_bfloat16_t));
    for (int i = 0; i < N; i++) c[i] = bfloat16_to_float(c16[i]);
    
    /* BFLOAT16 精度较低，使用较大容差 */
    for (int i = 0; i < 10; i++) ASSERT_NEAR(c[i], a[i] + b[i], 0.01f);
    
    ace_buffer_free(ba); ace_buffer_free(bb); ace_buffer_free(bc);
    return 1;
}

/* ============================================================================
 * INT8 测试
 * ============================================================================ */

static int test_vec_mul_int8(ace_device_t dev, ace_dtype_t dtype) {
    (void)dtype;
    const int N = 20;
    int8_t a[N], b[N], c[N];
    for (int i = 0; i < N; i++) { a[i] = (int8_t)(i % 10); b[i] = 2; }
    
    ace_buffer_t ba, bb, bc;
    ace_buffer_alloc(dev, N * sizeof(int8_t), &ba);
    ace_buffer_alloc(dev, N * sizeof(int8_t), &bb);
    ace_buffer_alloc(dev, N * sizeof(int8_t), &bc);
    
    ace_buffer_write(ba, a, N * sizeof(int8_t));
    ace_buffer_write(bb, b, N * sizeof(int8_t));
    
    int n = N;
    void* args[] = {&n, ba, bb, bc};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_vec_mul(), ACE_DTYPE_INT8, N, args, types, 4);
    ace_finish(dev);
    ace_buffer_read(bc, c, N * sizeof(int8_t));
    
    for (int i = 0; i < 10; i++) {
        if (c[i] != a[i] * b[i]) {
            printf("FAIL: %d != %d\n", (int)c[i], (int)(a[i] * b[i]));
            return 0;
        }
    }
    
    ace_buffer_free(ba); ace_buffer_free(bb); ace_buffer_free(bc);
    return 1;
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
    printf("Data Type Tests:\n");
    
    RUN_TEST(test_vec_add_float32, dev, ACE_DTYPE_FLOAT32, "FLOAT32 vec_add");
    RUN_TEST(test_vec_add_float16, dev, ACE_DTYPE_FLOAT16, "FLOAT16 vec_add");
    RUN_TEST(test_vec_add_bfloat16, dev, ACE_DTYPE_BFLOAT16, "BFLOAT16 vec_add");
    RUN_TEST(test_vec_mul_int8, dev, ACE_DTYPE_INT8, "INT8 vec_mul");
    
    ace_device_release(dev);
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main() {
    printf("========================================\n");
    printf("  AgierCompute - Data Type Tests\n");
    printf("========================================\n");

    /* CPU - 跳过，因为 CPU 后端是占位实现 */
    /* test_device("CPU", ACE_DEVICE_CPU, 0); */

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
    printf("  Results: %d/%d passed\n", g_passed, g_total);
    printf("========================================\n");
    
    return (g_passed == g_total) ? 0 : 1;
}
