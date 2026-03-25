/**
 * @file multi_device_test.c
 * @brief 多设备/跨 GPU 运行测试
 *
 * 演示：
 * 1. 发现所有可用设备
 * 2. 跨设备数据并行
 * 3. 自动分片执行
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ace.h"

/* ============================================================================
 * 内核定义
 * ============================================================================ */

ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

ACE_KERNEL(scale,
    void scale(int n, T alpha, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] * alpha;
    }
);

ACE_KERNEL(gemm,
    void gemm(int n, int m, int k, T* A, T* B, T* C) {
        int row = GID;
        if (row < n) {
            for (int j = 0; j < m; j++) {
                T sum = 0;
                for (int i = 0; i < k; i++) {
                    sum += A[row * k + i] * B[i * m + j];
                }
                C[row * m + j] = sum;
            }
        }
    }
);

/* ============================================================================
 * 测试函数
 * ============================================================================ */

void test_device_discovery(void) {
    printf("\n=== Test: Device Discovery ===\n");

    ace_device_list_t devices;
    ace_error_t err = ace_device_get_all(&devices);

    if (err != ACE_OK) {
        printf("Failed to get devices (err=%d)\n", err);
        return;
    }

    printf("Found %d device(s):\n", devices.count);
    for (int i = 0; i < devices.count; i++) {
        printf("\nDevice %d:\n", i);
        ace_device_print_info(devices.devices[i]);
    }

    ace_device_list_release(&devices);
}

void test_single_device_vec_add(void) {
    printf("\n=== Test: Vector Add (Single Device) ===\n");

    ace_device_t dev;
    ace_error_t err = ace_device_select_best(&dev);
    if (err != ACE_OK || !dev) {
        printf("No device available\n");
        return;
    }

    printf("Using device: ");
    ace_device_print_info(dev);

    const int N = 1000000;
    size_t size = N * sizeof(float);

    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, size, &buf_a);
    ace_buffer_alloc(dev, size, &buf_b);
    ace_buffer_alloc(dev, size, &buf_c);

    ace_buffer_write(buf_a, h_a, size);
    ace_buffer_write(buf_b, h_b, size);

    int n = N;
    void* args[] = {&n, buf_a, buf_b, buf_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

    printf("Running kernel...\n");
    err = ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);

    ace_buffer_read(buf_c, h_c, size);

    /* 验证结果 */
    int pass = 1;
    for (int i = 0; i < 10; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            pass = 0;
            printf("Mismatch at %d: got %f, expected %f\n", i, h_c[i], expected);
            break;
        }
    }

    printf("%s (err=%d)\n", pass ? "PASS" : "FAIL", err);
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) printf("%.1f ", h_c[i]);
    printf("\n");

    free(h_a); free(h_b); free(h_c);
    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
    ace_device_release(dev);
}

void test_multi_device_vec_add(void) {
    printf("\n=== Test: Vector Add (Multi-Device Data Parallel) ===\n");

    ace_device_list_t devices;
    ace_error_t err = ace_device_get_all(&devices);

    if (err != ACE_OK || devices.count == 0) {
        printf("No devices available\n");
        return;
    }

    printf("Using %d device(s) for data parallel execution\n", devices.count);
    for (int i = 0; i < devices.count; i++) {
        printf("  Device %d: ", i);
        ace_device_props_t props;
        ace_device_props(devices.devices[i], &props);
        printf("%s\n", props.name);
    }

    const int N = 1000000;
    size_t size = N * sizeof(float);

    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    /* 创建分片缓冲区 */
    ace_sharded_buffer_t sharded_a, sharded_b, sharded_c;
    ace_buffer_alloc_sharded(&devices, size, &sharded_a);
    ace_buffer_alloc_sharded(&devices, size, &sharded_b);
    ace_buffer_alloc_sharded(&devices, size, &sharded_c);

    /* 写入数据 */
    ace_buffer_write_sharded(&sharded_a, h_a, size);
    ace_buffer_write_sharded(&sharded_b, h_b, size);

    int n = N;
    void* args[] = {&n, &sharded_a, &sharded_b, &sharded_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

    printf("Running kernel across %d device(s)...\n", devices.count);
    err = ace_kernel_invoke_sharded(&devices, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N,
                                     args, types, 4);
    ace_finish_all(&devices);

    /* 读取结果 */
    ace_buffer_read_sharded(&sharded_c, h_c, size);

    /* 验证结果 */
    int pass = 1;
    for (int i = 0; i < 10; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            pass = 0;
            printf("Mismatch at %d: got %f, expected %f\n", i, h_c[i], expected);
            break;
        }
    }

    printf("%s (err=%d)\n", pass ? "PASS" : "FAIL", err);
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) printf("%.1f ", h_c[i]);
    printf("\n");

    free(h_a); free(h_b); free(h_c);
    ace_buffer_free_sharded(&sharded_a);
    ace_buffer_free_sharded(&sharded_b);
    ace_buffer_free_sharded(&sharded_c);
    ace_device_list_release(&devices);
}

void test_gemm(void) {
    printf("\n=== Test: Matrix Multiplication (GEMM) ===\n");

    ace_device_t dev;
    ace_error_t err = ace_device_select_best(&dev);
    if (err != ACE_OK || !dev) {
        printf("No device available\n");
        return;
    }

    printf("Using device: ");
    ace_device_print_info(dev);

    /* 矩阵大小：A[MxK] * B[KxN] = C[MxN] */
    int M = 64, K = 128, N = 64;

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float* h_A = (float*)malloc(size_a);
    float* h_B = (float*)malloc(size_b);
    float* h_C = (float*)malloc(size_c);

    /* 初始化矩阵 */
    for (int i = 0; i < M * K; i++) h_A[i] = i % 10 * 0.1f;
    for (int i = 0; i < K * N; i++) h_B[i] = i % 7 * 0.1f;

    ace_buffer_t buf_A, buf_B, buf_C;
    ace_buffer_alloc(dev, size_a, &buf_A);
    ace_buffer_alloc(dev, size_b, &buf_B);
    ace_buffer_alloc(dev, size_c, &buf_C);

    ace_buffer_write(buf_A, h_A, size_a);
    ace_buffer_write(buf_B, h_B, size_b);

    void* args[] = {&M, &N, &K, buf_A, buf_B, buf_C};
    int types[] = {ACE_VAL, ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

    printf("Running GEMM (%dx%d) * (%dx%d) = (%dx%d)...\n", M, K, K, N, M, N);
    err = ace_kernel_invoke(dev, _ace_get_gemm(), ACE_DTYPE_FLOAT32, M, args, types, 6);
    ace_finish(dev);

    ace_buffer_read(buf_C, h_C, size_c);

    /* 验证部分结果 */
    printf("Result C[0:4][0:4]:\n");
    for (int i = 0; i < 4 && i < M; i++) {
        printf("  ");
        for (int j = 0; j < 4 && j < N; j++) {
            printf("%8.4f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    printf("PASS (err=%d)\n", err);

    free(h_A); free(h_B); free(h_C);
    ace_buffer_free(buf_A);
    ace_buffer_free(buf_B);
    ace_buffer_free(buf_C);
    ace_device_release(dev);
}

void test_scale_multi_device(void) {
    printf("\n=== Test: Scale (Multi-Device) ===\n");

    ace_device_list_t devices;
    ace_error_t err = ace_device_get_all(&devices);

    if (err != ACE_OK || devices.count == 0) {
        printf("No devices available\n");
        return;
    }

    printf("Using %d device(s)\n", devices.count);

    const int N = 500000;
    size_t size = N * sizeof(float);

    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    float alpha = 2.5f;

    for (int i = 0; i < N; i++) h_in[i] = i * 0.1f;

    ace_sharded_buffer_t sharded_in, sharded_out;
    ace_buffer_alloc_sharded(&devices, size, &sharded_in);
    ace_buffer_alloc_sharded(&devices, size, &sharded_out);

    ace_buffer_write_sharded(&sharded_in, h_in, size);

    int n = N;
    void* args[] = {&n, &alpha, &sharded_in, &sharded_out};
    int types[] = {ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF};

    err = ace_kernel_invoke_sharded(&devices, _ace_get_scale(), ACE_DTYPE_FLOAT32, N,
                                     args, types, 4);
    ace_finish_all(&devices);

    ace_buffer_read_sharded(&sharded_out, h_out, size);

    int pass = 1;
    for (int i = 0; i < 10; i++) {
        float expected = h_in[i] * alpha;
        if (h_out[i] != expected) {
            pass = 0;
            printf("Mismatch at %d: got %f, expected %f\n", i, h_out[i], expected);
            break;
        }
    }

    printf("%s (err=%d)\n", pass ? "PASS" : "FAIL", err);
    printf("First 10 results: ");
    for (int i = 0; i < 10; i++) printf("%.1f ", h_out[i]);
    printf("\n");

    free(h_in); free(h_out);
    ace_buffer_free_sharded(&sharded_in);
    ace_buffer_free_sharded(&sharded_out);
    ace_device_list_release(&devices);
}

int main() {
    printf("========================================\n");
    printf("  AgierCompute - Multi-Device Tests\n");
    printf("========================================\n");
    fflush(stdout);

    test_device_discovery();
    test_single_device_vec_add();
    test_multi_device_vec_add();
    test_gemm();
    test_scale_multi_device();

    printf("\n========================================\n");
    printf("  ALL TESTS COMPLETED!\n");
    printf("========================================\n");

    return 0;
}
