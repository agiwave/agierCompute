/**
 * @file vulkan_test.c
 * @brief Vulkan 后端测试
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ace.h"

/* 内核必须在文件作用域定义 */
ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
)

int test_device(ace_device_t dev, int idx) {
    ace_device_props_t props;
    ace_device_props(dev, &props);
    printf("Device %d: %s\n", idx, props.name);

    const int N = 100;
    float *h_a = malloc(N * sizeof(float));
    float *h_b = malloc(N * sizeof(float));
    float *h_c = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_a);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_b);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_c);

    ace_buffer_write(buf_a, h_a, N * sizeof(float));
    ace_buffer_write(buf_b, h_b, N * sizeof(float));

    int n = N;
    void* args[] = {&n, buf_a, buf_b, buf_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

    printf("  Running kernel... ");
    fflush(stdout);

    ace_error_t err = ACE_INVOKE(dev, vec_add, ACE_DTYPE_FLOAT32, N, &N, buf_a, buf_b, buf_c);

    if (err != ACE_OK) {
        printf("FAILED (err=%d)\n", err);
        ace_finish(dev);
        free(h_a); free(h_b); free(h_c);
        ace_buffer_free(buf_a);
        ace_buffer_free(buf_b);
        ace_buffer_free(buf_c);
        ace_device_release(dev);
        return 0;
    }

    printf("OK\n");
    ace_finish(dev);
    ace_error_t read_err = ace_buffer_read(buf_c, h_c, N * sizeof(float));
    if (read_err != ACE_OK) {
        printf("  Read failed: %d\n", read_err);
        free(h_a); free(h_b); free(h_c);
        ace_buffer_free(buf_a);
        ace_buffer_free(buf_b);
        ace_buffer_free(buf_c);
        ace_device_release(dev);
        return 0;
    }

    int pass = 1;
    for (int i = 0; i < 10; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) { pass = 0; break; }
    }
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");

    free(h_a); free(h_b); free(h_c);
    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
    ace_device_release(dev);
    return pass;
}

int main() {
    printf("========================================\n");
    printf("  Vulkan Backend Tests\n");
    printf("========================================\n\n");

    int count = 0;
    ace_device_count(ACE_DEVICE_VULKAN, &count);
    printf("Vulkan devices: %d\n\n", count);

    if (count == 0) {
        printf("No Vulkan device\n");
        return 1;
    }

    int passed = 0;
    for (int idx = 0; idx < count; idx++) {
        ace_device_t dev;
        ace_error_t err = ace_device_get(ACE_DEVICE_VULKAN, idx, &dev);
        if (err != ACE_OK || !dev) {
            printf("Device %d: Failed to get device\n\n", idx);
            continue;
        }

        if (test_device(dev, idx)) passed++;
        printf("\n");
    }

    printf("========================================\n");
    printf("  Results: %d/%d devices passed\n", passed, count);
    printf("========================================\n");
    return (passed == count) ? 0 : 1;
}
