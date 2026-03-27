/**
 * @file user_kernels.c
 * @brief 用户自定义内核示例 - 展示更多内核类型
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"

/* ============================================================================
 * 内核定义
 * ============================================================================ */

ACE_KERNEL(kernel_vec_mul,
    void vec_mul(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] * b[i];
    }
)

ACE_KERNEL(kernel_sigmoid,
    void sigmoid(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = 1.0 / (1.0 + exp(-in[i]));
    }
)

ACE_KERNEL(kernel_fill,
    void fill(int n, T val, T* out) {
        int i = GID;
        if (i < n) out[i] = val;
    }
)

ACE_KERNEL(kernel_sqrt,
    void kernel_sqrt(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = sqrt(in[i]);
    }
)

ACE_KERNEL(kernel_exp,
    void kernel_exp(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = exp(in[i]);
    }
)

/* ============================================================================
 * 测试函数
 * ============================================================================ */

static void test_vec_mul(ace_device_t dev) {
    printf("\n--- Test: vec_mul ---\n");

    const int N = 6;
    int n = N;
    float h_a[] = {1, 2, 3, 4, 5, 6};
    float h_b[] = {2, 3, 4, 5, 6, 7};
    float h_c[6];

    ace_buffer_t buf_a, buf_b, buf_c;
    ACE_CHECK_VOID(ace_buffer_alloc(dev, N * sizeof(float), &buf_a));
    ACE_CHECK_VOID(ace_buffer_alloc(dev, N * sizeof(float), &buf_b));
    ACE_CHECK_VOID(ace_buffer_alloc(dev, N * sizeof(float), &buf_c));

    ACE_CHECK_VOID(ace_buffer_write(buf_a, h_a, N * sizeof(float)));
    ACE_CHECK_VOID(ace_buffer_write(buf_b, h_b, N * sizeof(float)));

    /* 使用简化宏 */
    ACE_INVOKE(dev, kernel_vec_mul, ACE_DTYPE_FLOAT32, N, &n, buf_a, buf_b, buf_c);
    ace_finish(dev);

    ACE_CHECK_VOID(ace_buffer_read(buf_c, h_c, N * sizeof(float)));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - h_a[i] * h_b[i]) > 0.001f) pass = 0;
    }
    printf("%s\n", pass ? "PASS" : "FAIL");
    printf("  Results: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_c[i]);
    printf("\n");

    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
}

static void test_sigmoid(ace_device_t dev) {
    printf("\n--- Test: sigmoid ---\n");

    const int N = 5;
    float h_in[] = {-2, -1, 0, 1, 2};
    float h_out[5];

    ace_buffer_t buf_in, buf_out;
    ACE_CHECK_VOID(ace_buffer_alloc(dev, N * sizeof(float), &buf_in));
    ACE_CHECK_VOID(ace_buffer_alloc(dev, N * sizeof(float), &buf_out));
    ACE_CHECK_VOID(ace_buffer_write(buf_in, h_in, N * sizeof(float)));

    /* 使用简化宏 */
    int n = N;
    ACE_INVOKE(dev, kernel_sigmoid, ACE_DTYPE_FLOAT32, N, &n, buf_in, buf_out);
    ace_finish(dev);

    ACE_CHECK_VOID(ace_buffer_read(buf_out, h_out, N * sizeof(float)));

    printf("  Input:  ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_in[i]);
    printf("\n  Output: ");
    for (int i = 0; i < N; i++) printf("%.3f ", h_out[i]);
    printf("\n");

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
}

static void test_fill(ace_device_t dev) {
    printf("\n--- Test: fill ---\n");

    const int N = 5;
    float h_out[5] = {0};
    float fill_val = 3.14159f;

    ace_buffer_t buf_out;
    ACE_CHECK_VOID(ace_buffer_alloc(dev, N * sizeof(float), &buf_out));

    /* 多标量参数，使用原始 API */
    int n = N;
    void* args[] = {&n, &fill_val, buf_out};
    int sizes[] = {sizeof(int), sizeof(float), 0};
    ace_kernel_invoke(dev, _ace_get_kernel_fill(), ACE_DTYPE_FLOAT32, N, args, sizes, 3);
    ace_finish(dev);

    ACE_CHECK_VOID(ace_buffer_read(buf_out, h_out, N * sizeof(float)));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (fabs(h_out[i] - fill_val) > 0.001f) pass = 0;
    }
    printf("%s\n", pass ? "PASS" : "FAIL");
    printf("  Results: ");
    for (int i = 0; i < N; i++) printf("%.2f ", h_out[i]);
    printf("\n");

    ace_buffer_free(buf_out);
}

static void test_sqrt(ace_device_t dev) {
    printf("\n--- Test: sqrt ---\n");

    const int N = 5;
    int n = N;
    float h_in[] = {1, 4, 9, 16, 25};
    float h_out[5];

    ace_buffer_t buf_in, buf_out;
    ACE_CHECK_VOID(ace_buffer_alloc(dev, N * sizeof(float), &buf_in));
    ACE_CHECK_VOID(ace_buffer_alloc(dev, N * sizeof(float), &buf_out));
    ACE_CHECK_VOID(ace_buffer_write(buf_in, h_in, N * sizeof(float)));

    /* 使用简化宏 */
    ACE_INVOKE(dev, kernel_sqrt, ACE_DTYPE_FLOAT32, N, &n, buf_in, buf_out);
    ace_finish(dev);

    ACE_CHECK_VOID(ace_buffer_read(buf_out, h_out, N * sizeof(float)));

    printf("  Input:  ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_in[i]);
    printf("\n  Output: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_out[i]);
    printf("\n");

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
}

static void test_exp(ace_device_t dev) {
    printf("\n--- Test: exp ---\n");

    const int N = 4;
    int n = N;
    float h_in[] = {0, 1, 2, 3};
    float h_out[4];

    ace_buffer_t buf_in, buf_out;
    ACE_CHECK_VOID(ace_buffer_alloc(dev, N * sizeof(float), &buf_in));
    ACE_CHECK_VOID(ace_buffer_alloc(dev, N * sizeof(float), &buf_out));
    ACE_CHECK_VOID(ace_buffer_write(buf_in, h_in, N * sizeof(float)));

    /* 使用简化宏 */
    ACE_INVOKE(dev, kernel_exp, ACE_DTYPE_FLOAT32, N, &n, buf_in, buf_out);
    ace_finish(dev);

    ACE_CHECK_VOID(ace_buffer_read(buf_out, h_out, N * sizeof(float)));

    printf("  Input:  ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_in[i]);
    printf("\n  Output: ");
    for (int i = 0; i < N; i++) printf("%.3f ", h_out[i]);
    printf("\n");

    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
}

int main() {
    printf("========================================\n");
    printf("  AgierCompute - User Kernels Demo\n");
    printf("========================================\n");
    fflush(stdout);

    /* 获取 Vulkan 设备（优先使用，因为更稳定） */
    int count = 0;
    ace_device_count(ACE_DEVICE_VULKAN, &count);
    printf("Vulkan devices: %d\n", count);

    if (count > 0) {
        ace_device_t dev = NULL;
        ace_error_t err = ace_device_get(ACE_DEVICE_VULKAN, 0, &dev);

        if (err == ACE_OK && dev) {
            ace_device_props_t props;
            ace_device_props(dev, &props);
            printf("Using: %s\n", props.name);

            /* 运行所有测试 */
            test_vec_mul(dev);
            test_sigmoid(dev);
            test_fill(dev);
            test_sqrt(dev);
            test_exp(dev);

            ace_device_release(dev);
        } else {
            printf("Failed to get device (err=%d)\n", err);
        }
    }

    printf("\n========================================\n");
    printf("  All tests completed!\n");
    printf("========================================\n");

    return 0;
}
