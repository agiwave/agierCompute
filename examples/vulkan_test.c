/**
 * @file vulkan_test.c
 * @brief Vulkan 后端测试
 */
#include <stdio.h>
#include <stdlib.h>
#include "ace.h"

ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
);

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

    for (int idx = 0; idx < count; idx++) {
        ace_device_t dev;
        ace_device_get(ACE_DEVICE_VULKAN, idx, &dev);

        ace_device_props_t props;
        ace_device_props(dev, &props);
        printf("Device %d: %s\n", idx, props.name);
        printf("  Compute units: %d\n", props.compute_units);
        printf("  Memory: %zu MB\n\n", props.total_memory / (1024*1024));
        
        /* 跳过 Intel 集成显卡（已知问题） */
        if (strstr(props.name, "Intel") != NULL) {
            printf("  Skipping Intel device (known issue)\n\n");
            ace_device_release(dev);
            continue;
        }

        /* Test: vec_add */
        const int N = 1000;
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

        printf("Running kernel...\n");
        ace_error_t err = ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
        ace_finish(dev);
        ace_buffer_read(buf_c, h_c, N * sizeof(float));

        int pass = 1;
        for (int i = 0; i < 10; i++) {
            if (h_c[i] != h_a[i] + h_b[i]) { pass = 0; break; }
        }
        printf("%s (err=%d)\n", pass ? "PASS" : "FAIL", err);
        printf("First 10: ");
        for (int i = 0; i < 10; i++) printf("%.0f ", h_c[i]);
        printf("\n\n");

        free(h_a); free(h_b); free(h_c);
        ace_buffer_free(buf_a);
        ace_buffer_free(buf_b);
        ace_buffer_free(buf_c);
        ace_device_release(dev);
    }

    printf("========================================\n");
    printf("  Vulkan Tests Completed!\n");
    printf("========================================\n");
    return 0;
}
