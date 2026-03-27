/**
 * @file test_backend.c
 * @brief 后端基础测试 - 统一测试所有 GPU 后端
 *
 * 测试内容：
 * - 设备获取和属性查询
 * - 内存分配/写入/读取
 * - 内核编译和执行
 * - 基础向量运算
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"

/* 内核定义 */
ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
)

ACE_KERNEL(vec_scale,
    void vec_scale(int n, T alpha, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] * alpha;
    }
)

/* 测试统计 */
static int g_passed = 0;
static int g_failed = 0;

/* ============================================================================
 * 测试函数
 * ============================================================================ */

static int test_device_props(ace_device_t dev) {
    ace_device_props_t props;
    ace_error_t err = ace_device_props(dev, &props);
    if (err != ACE_OK) return 0;
    
    /* 验证属性有效性 */
    if (props.name[0] == '\0') return 0;
    if (props.total_memory == 0) return 0;
    
    return 1;
}

static int test_memory_ops(ace_device_t dev) {
    const int N = 64;
    float h_a[N], h_b[N], h_c[N];
    
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    ace_buffer_t buf_a, buf_b, buf_c;
    
    /* 分配 */
    if (ace_buffer_alloc(dev, N * sizeof(float), &buf_a) != ACE_OK) return 0;
    if (ace_buffer_alloc(dev, N * sizeof(float), &buf_b) != ACE_OK) {
        ace_buffer_free(buf_a);
        return 0;
    }
    if (ace_buffer_alloc(dev, N * sizeof(float), &buf_c) != ACE_OK) {
        ace_buffer_free(buf_a);
        ace_buffer_free(buf_b);
        return 0;
    }
    
    /* 写入 */
    if (ace_buffer_write(buf_a, h_a, N * sizeof(float)) != ACE_OK) {
        ace_buffer_free(buf_a);
        ace_buffer_free(buf_b);
        ace_buffer_free(buf_c);
        return 0;
    }
    if (ace_buffer_write(buf_b, h_b, N * sizeof(float)) != ACE_OK) {
        ace_buffer_free(buf_a);
        ace_buffer_free(buf_b);
        ace_buffer_free(buf_c);
        return 0;
    }
    
    /* 执行内核 */
    int n = N;
    ACE_INVOKE(dev, vec_add, ACE_DTYPE_FLOAT32, N, n, buf_a, buf_b, buf_c);
    ace_finish(dev);
    
    /* 读取 */
    if (ace_buffer_read(buf_c, h_c, N * sizeof(float)) != ACE_OK) {
        ace_buffer_free(buf_a);
        ace_buffer_free(buf_b);
        ace_buffer_free(buf_c);
        return 0;
    }
    
    /* 验证 */
    int ok = 1;
    for (int i = 0; i < N && ok; i++) {
        if (fabsf(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) ok = 0;
    }
    
    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
    
    return ok;
}

static int test_scalar_param(ace_device_t dev) {
    const int N = 32;
    float h_in[N], h_out[N];
    float alpha = 2.5f;
    
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i * 0.1f;
    }
    
    ace_buffer_t buf_in, buf_out;
    if (ace_buffer_alloc(dev, N * sizeof(float), &buf_in) != ACE_OK) return 0;
    if (ace_buffer_alloc(dev, N * sizeof(float), &buf_out) != ACE_OK) {
        ace_buffer_free(buf_in);
        return 0;
    }
    
    ace_buffer_write(buf_in, h_in, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, &alpha, buf_in, buf_out};
    int sizes[] = {sizeof(int), sizeof(float), 0, 0};
    
    if (ace_kernel_invoke(dev, _ace_get_vec_scale(), ACE_DTYPE_FLOAT32, N, args, sizes, 4) != ACE_OK) {
        ace_buffer_free(buf_in);
        ace_buffer_free(buf_out);
        return 0;
    }
    ace_finish(dev);
    ace_buffer_read(buf_out, h_out, N * sizeof(float));
    
    int ok = 1;
    for (int i = 0; i < N && ok; i++) {
        float expected = h_in[i] * alpha;
        if (fabsf(h_out[i] - expected) > 1e-4f) ok = 0;
    }
    
    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
    
    return ok;
}

/* ============================================================================
 * 后端测试
 * ============================================================================ */

static void test_backend(const char* name, ace_device_type_t type) {
    int count = 0;
    ace_device_count(type, &count);
    
    if (count == 0) {
        printf("\n[%s] No devices found\n", name);
        return;
    }
    
    printf("\n========================================\n");
    printf(" %s Backend (%d device%s)\n", name, count, count > 1 ? "s" : "");
    printf("========================================\n");
    
    for (int idx = 0; idx < count; idx++) {
        ace_device_t dev;
        if (ace_device_get(type, idx, &dev) != ACE_OK || !dev) {
            printf("  Device %d: Failed to get\n", idx);
            continue;
        }
        
        ace_device_props_t props;
        ace_device_props(dev, &props);
        printf("\nDevice %d: %s\n", idx, props.name);
        printf("  Memory: %zu MB\n", props.total_memory / (1024 * 1024));
        
        /* 测试 1: 设备属性 */
        printf("  [1/3] Device props ... ");
        if (test_device_props(dev)) {
            printf("PASS\n");
            g_passed++;
        } else {
            printf("FAIL\n");
            g_failed++;
        }
        
        /* 测试 2: 内存操作 */
        printf("  [2/3] Memory ops   ... ");
        if (test_memory_ops(dev)) {
            printf("PASS\n");
            g_passed++;
        } else {
            printf("FAIL\n");
            g_failed++;
        }
        
        /* 测试 3: 标量参数 */
        printf("  [3/3] Scalar param ... ");
        if (test_scalar_param(dev)) {
            printf("PASS\n");
            g_passed++;
        } else {
            printf("FAIL\n");
            g_failed++;
        }
        
        ace_device_release(dev);
    }
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main() {
    printf("========================================================\n");
    printf("  AgierCompute - Backend Test Suite\n");
    printf("========================================================\n");
    
    /* 测试所有 GPU 后端 */
    test_backend("CUDA", ACE_DEVICE_CUDA);
    test_backend("OpenCL", ACE_DEVICE_OPENCL);
    test_backend("Vulkan", ACE_DEVICE_VULKAN);
    
    printf("\n========================================================\n");
    printf("  Results: %d passed, %d failed\n", g_passed, g_failed);
    printf("========================================================\n");
    
    return (g_failed > 0) ? 1 : 0;
}
