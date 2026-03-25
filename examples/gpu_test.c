/**
 * @file gpu_test.c
 * @brief GPU 后端测试 - 只测试 CUDA/OpenCL/Vulkan
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ace.h"

ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

void test_gpu_backend(const char* name, ace_device_type_t type) {
    printf("\n=== Testing %s Backend ===\n", name);

    int count = 0;
    ace_error_t err = ace_device_count(type, &count);
    printf("%s devices: %d\n", name, count);

    if (count == 0) {
        printf("No %s device available\n", name);
        return;
    }

    ace_device_t dev = NULL;
    err = ace_device_get(type, 0, &dev);
    if (err != ACE_OK || !dev) {
        printf("Failed to get device (err=%d)\n", err);
        return;
    }

    ace_device_props_t props;
    ace_device_props(dev, &props);
    printf("Device: %s\n", props.name);
    printf("  Vendor: %s\n", props.vendor);
    printf("  Compute units: %d\n", props.compute_units);

    /* 测试向量加法 */
    const int N = 100;
    float h_a[N], h_b[N], h_c[N];
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

    printf("Running kernel...\n");
    err = ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    if (err != ACE_OK) {
        printf("Kernel invoke failed (err=%d)\n", err);
    } else {
        ace_finish(dev);
        ace_buffer_read(buf_c, h_c, N * sizeof(float));

        int pass = 1;
        for (int i = 0; i < 10; i++) {
            if (h_c[i] != h_a[i] + h_b[i]) {
                pass = 0;
                break;
            }
        }
        printf("%s (err=%d)\n", pass ? "PASS" : "FAIL", err);
        printf("Results: ");
        for (int i = 0; i < 10; i++) printf("%.0f ", h_c[i]);
        printf("\n");
    }

    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
    ace_device_release(dev);
}

int main() {
    printf("========================================\n");
    printf("  AgierCompute - GPU Backend Tests\n");
    printf("========================================\n");
    fflush(stdout);

    /* 只测试 GPU 后端 */
    test_gpu_backend("CUDA", ACE_DEVICE_CUDA);
    test_gpu_backend("OpenCL", ACE_DEVICE_OPENCL);
    test_gpu_backend("Vulkan", ACE_DEVICE_VULKAN);

    printf("\n========================================\n");
    printf("  GPU Tests Completed!\n");
    printf("========================================\n");

    return 0;
}
