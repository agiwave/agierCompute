/**
 * @file kernel_tests.c
 * @brief 内核测试 - 用户定义内核示例
 * 
 * 展示用户如何使用 ACE_KERNEL 宏定义自己的内核
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"

/* ============================================================================
 * 用户定义内核示例
 * ============================================================================ */

/* 向量加法 */
ACE_KERNEL(my_vec_add,
    void my_vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

/* 向量乘法 */
ACE_KERNEL(my_vec_mul,
    void my_vec_mul(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] * b[i];
    }
);

/* 缩放 */
ACE_KERNEL(my_scale,
    void my_scale(int n, T alpha, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] * alpha;
    }
);

/* ReLU 激活 */
ACE_KERNEL(my_relu,
    void my_relu(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] > 0 ? in[i] : 0;
    }
);

/* Sigmoid 激活 */
ACE_KERNEL(my_sigmoid,
    void my_sigmoid(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = 1.0 / (1.0 + exp(-in[i]));
    }
);

/* 绝对值 */
ACE_KERNEL(my_abs,
    void my_abs(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] < 0 ? -in[i] : in[i];
    }
);

/* 平方 */
ACE_KERNEL(my_square,
    void my_square(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] * in[i];
    }
);

/* 填充常数 */
ACE_KERNEL(my_fill,
    void my_fill(int n, T val, T* out) {
        int i = GID;
        if (i < n) out[i] = val;
    }
);

/* 拷贝 */
ACE_KERNEL(my_copy,
    void my_copy(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i];
    }
);

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

#define RUN_TEST(fn, dev, name) do { \
    g_total++; \
    printf("  %-15s ... ", name); \
    if (fn(dev)) { g_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

/* ============================================================================
 * 测试函数
 * ============================================================================ */

static int test_vec_add(ace_device_t dev) {
    const int N = 20;
    float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) { a[i] = i; b[i] = i * 2; }
    
    ace_buffer_t ba, bb, bc;
    ace_buffer_alloc(dev, N * sizeof(float), &ba);
    ace_buffer_alloc(dev, N * sizeof(float), &bb);
    ace_buffer_alloc(dev, N * sizeof(float), &bc);
    
    ace_buffer_write(ba, a, N * sizeof(float));
    ace_buffer_write(bb, b, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, ba, bb, bc};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_my_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);
    ace_buffer_read(bc, c, N * sizeof(float));
    
    for (int i = 0; i < 10; i++) ASSERT_NEAR(c[i], a[i] + b[i], 1e-5f);
    
    ace_buffer_free(ba); ace_buffer_free(bb); ace_buffer_free(bc);
    return 1;
}

static int test_vec_mul(ace_device_t dev) {
    const int N = 20;
    float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) { a[i] = i * 0.5f; b[i] = i * 2.0f; }
    
    ace_buffer_t ba, bb, bc;
    ace_buffer_alloc(dev, N * sizeof(float), &ba);
    ace_buffer_alloc(dev, N * sizeof(float), &bb);
    ace_buffer_alloc(dev, N * sizeof(float), &bc);
    ace_buffer_write(ba, a, N * sizeof(float));
    ace_buffer_write(bb, b, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, ba, bb, bc};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_my_vec_mul(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);
    ace_buffer_read(bc, c, N * sizeof(float));
    
    for (int i = 0; i < 10; i++) ASSERT_NEAR(c[i], a[i] * b[i], 1e-4f);
    
    ace_buffer_free(ba); ace_buffer_free(bb); ace_buffer_free(bc);
    return 1;
}

static int test_scale(ace_device_t dev) {
    const int N = 20;
    float in[N], out[N], alpha = 2.5f;
    for (int i = 0; i < N; i++) in[i] = i * 0.1f;
    
    ace_buffer_t bin, bout;
    ace_buffer_alloc(dev, N * sizeof(float), &bin);
    ace_buffer_alloc(dev, N * sizeof(float), &bout);
    ace_buffer_write(bin, in, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, &alpha, bin, bout};
    int types[] = {ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_my_scale(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);
    ace_buffer_read(bout, out, N * sizeof(float));
    
    for (int i = 0; i < 10; i++) ASSERT_NEAR(out[i], in[i] * alpha, 1e-5f);
    
    ace_buffer_free(bin); ace_buffer_free(bout);
    return 1;
}

static int test_relu(ace_device_t dev) {
    const int N = 10;
    float in[] = {-5, -2, -1, 0, 1, 2, 3, 4, 5, 10};
    float expected[] = {0, 0, 0, 0, 1, 2, 3, 4, 5, 10};
    float out[N];
    
    ace_buffer_t bin, bout;
    ace_buffer_alloc(dev, N * sizeof(float), &bin);
    ace_buffer_alloc(dev, N * sizeof(float), &bout);
    ace_buffer_write(bin, in, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, bin, bout};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_my_relu(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(bout, out, N * sizeof(float));
    
    for (int i = 0; i < N; i++) ASSERT_NEAR(out[i], expected[i], 1e-5f);
    
    ace_buffer_free(bin); ace_buffer_free(bout);
    return 1;
}

static int test_sigmoid(ace_device_t dev) {
    const int N = 5;
    float in[] = {-2, -1, 0, 1, 2};
    float expected[] = {0.1192, 0.2689, 0.5, 0.7311, 0.8808};
    float out[N];
    
    ace_buffer_t bin, bout;
    ace_buffer_alloc(dev, N * sizeof(float), &bin);
    ace_buffer_alloc(dev, N * sizeof(float), &bout);
    ace_buffer_write(bin, in, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, bin, bout};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_my_sigmoid(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(bout, out, N * sizeof(float));
    
    for (int i = 0; i < N; i++) ASSERT_NEAR(out[i], expected[i], 0.01f);
    
    ace_buffer_free(bin); ace_buffer_free(bout);
    return 1;
}

static int test_abs(ace_device_t dev) {
    const int N = 10;
    float in[] = {-5, -2, 0, 2, 5, -10, 10, -1, 1, 0};
    float expected[] = {5, 2, 0, 2, 5, 10, 10, 1, 1, 0};
    float out[N];
    
    ace_buffer_t bin, bout;
    ace_buffer_alloc(dev, N * sizeof(float), &bin);
    ace_buffer_alloc(dev, N * sizeof(float), &bout);
    ace_buffer_write(bin, in, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, bin, bout};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_my_abs(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(bout, out, N * sizeof(float));
    
    for (int i = 0; i < N; i++) ASSERT_NEAR(out[i], expected[i], 1e-5f);
    
    ace_buffer_free(bin); ace_buffer_free(bout);
    return 1;
}

static int test_square(ace_device_t dev) {
    const int N = 10;
    float in[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float expected[] = {0, 1, 4, 9, 16, 25, 36, 49, 64, 81};
    float out[N];
    
    ace_buffer_t bin, bout;
    ace_buffer_alloc(dev, N * sizeof(float), &bin);
    ace_buffer_alloc(dev, N * sizeof(float), &bout);
    ace_buffer_write(bin, in, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, bin, bout};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_my_square(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(bout, out, N * sizeof(float));
    
    for (int i = 0; i < N; i++) ASSERT_NEAR(out[i], expected[i], 1e-5f);
    
    ace_buffer_free(bin); ace_buffer_free(bout);
    return 1;
}

static int test_fill(ace_device_t dev) {
    const int N = 10;
    float val = 3.14159f, out[N];
    
    ace_buffer_t bout;
    ace_buffer_alloc(dev, N * sizeof(float), &bout);
    
    int n = N;
    void* args[] = {&n, &val, bout};
    int types[] = {ACE_VAL, ACE_VAL, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_my_fill(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(bout, out, N * sizeof(float));
    
    for (int i = 0; i < N; i++) ASSERT_NEAR(out[i], val, 1e-5f);
    
    ace_buffer_free(bout);
    return 1;
}

static int test_copy(ace_device_t dev) {
    const int N = 10;
    float in[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float out[N];
    
    ace_buffer_t bin, bout;
    ace_buffer_alloc(dev, N * sizeof(float), &bin);
    ace_buffer_alloc(dev, N * sizeof(float), &bout);
    ace_buffer_write(bin, in, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, bin, bout};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_kernel_invoke(dev, _ace_get_my_copy(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    ace_buffer_read(bout, out, N * sizeof(float));
    
    for (int i = 0; i < N; i++) ASSERT_NEAR(out[i], in[i], 1e-5f);
    
    ace_buffer_free(bin); ace_buffer_free(bout);
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
    
    printf("User Kernels:\n");
    RUN_TEST(test_vec_add, dev, "vec_add");
    RUN_TEST(test_vec_mul, dev, "vec_mul");
    RUN_TEST(test_scale, dev, "scale");
    RUN_TEST(test_relu, dev, "relu");
    RUN_TEST(test_sigmoid, dev, "sigmoid");
    RUN_TEST(test_abs, dev, "abs");
    RUN_TEST(test_square, dev, "square");
    RUN_TEST(test_fill, dev, "fill");
    RUN_TEST(test_copy, dev, "copy");
    
    ace_device_release(dev);
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main() {
    printf("========================================\n");
    printf("  AgierCompute - User Kernel Tests\n");
    printf("========================================\n");
    
    /* CPU */
    test_device("CPU", ACE_DEVICE_CPU, 0);
    
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
