/**
 * @file simple_test.c
 * @brief 简单测试 - 验证基本功能
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ace.h"

/* 内核定义 - 必须在函数外部 */
ACE_KERNEL(test_vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
)

int main() {
    printf("========================================\n");
    printf("  AgierCompute - Simple Test\n");
    printf("========================================\n");
    fflush(stdout);

    /* 测试 CPU 后端 */
    printf("\n--- Testing CPU Backend ---\n");

    int count = 0;
    ace_error_t err = ace_device_count(ACE_DEVICE_CPU, &count);
    printf("CPU devices: %d (err=%d)\n", count, err);

    if (count > 0) {
        ace_device_t dev = NULL;
        err = ace_device_get(ACE_DEVICE_CPU, 0, &dev);

        if (err == ACE_OK && dev) {
            /* 获取设备属性 */
            ace_device_props_t props;
            ace_device_props(dev, &props);
            printf("Device: %s\n", props.name);
            printf("Threads: %d\n", props.compute_units);

            /* 测试向量加法 */
            const int N = 8;
            float h_a[] = {1, 2, 3, 4, 5, 6, 7, 8};
            float h_b[] = {10, 20, 30, 40, 50, 60, 70, 80};
            float h_c[8];

            ace_buffer_t buf_a, buf_b, buf_c;
            ACE_CHECK(ace_buffer_alloc(dev, N * sizeof(float), &buf_a));
            ACE_CHECK(ace_buffer_alloc(dev, N * sizeof(float), &buf_b));
            ACE_CHECK(ace_buffer_alloc(dev, N * sizeof(float), &buf_c));

            ACE_CHECK(ace_buffer_write(buf_a, h_a, N * sizeof(float)));
            ACE_CHECK(ace_buffer_write(buf_b, h_b, N * sizeof(float)));

            /* 使用简化宏执行内核 */
            int n = N;
            ACE_INVOKE_1D(dev, test_vec_add, FLOAT32, N, &n, buf_a, buf_b, buf_c);
            ace_finish(dev);

            ACE_CHECK(ace_buffer_read(buf_c, h_c, N * sizeof(float)));

            printf("vec_add result: ");
            for (int i = 0; i < N; i++) printf("%.0f ", h_c[i]);
            printf("\n");

            /* 验证结果 */
            int pass = 1;
            for (int i = 0; i < N; i++) {
                if (h_c[i] != h_a[i] + h_b[i]) {
                    pass = 0;
                    break;
                }
            }
            printf("Test: %s\n", pass ? "PASS" : "FAIL");

            ace_buffer_free(buf_a);
            ace_buffer_free(buf_b);
            ace_buffer_free(buf_c);
            ace_device_release(dev);
        } else {
            printf("Failed to get device (err=%d)\n", err);
        }
    }

    printf("\n========================================\n");
    printf("  Test completed!\n");
    printf("========================================\n");

    return 0;
}
