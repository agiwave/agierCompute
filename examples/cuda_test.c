/**
 * @file cuda_test.c
 * @brief CUDA 后端测试
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

ACE_KERNEL(scale,
    void scale(int n, T alpha, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] * alpha;
    }
);

int main() {
    printf("========================================\n");
    printf("  CUDA Backend Tests\n");
    printf("========================================\n\n");

    int count = 0;
    ace_device_count(ACE_DEVICE_CUDA, &count);
    printf("CUDA devices: %d\n\n", count);

    if (count == 0) {
        printf("No CUDA device\n");
        return 1;
    }

    ace_device_t dev;
    ace_device_get(ACE_DEVICE_CUDA, 0, &dev);

    ace_device_props_t props;
    ace_device_props(dev, &props);
    printf("Device: %s\n", props.name);
    printf("  Compute units: %d\n", props.compute_units);
    printf("  Max threads: %zu\n", props.max_threads);
    printf("  Memory: %zu MB\n\n", props.total_memory / (1024*1024));

    /* Test 1: vec_add */
    printf("--- Test: vec_add (float) ---\n");
    {
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
    }

    /* Test 2: scale */
    printf("--- Test: scale (float) ---\n");
    {
        const int N = 100;
        float *h_in = malloc(N * sizeof(float));
        float *h_out = malloc(N * sizeof(float));
        float alpha = 2.5f;

        for (int i = 0; i < N; i++) h_in[i] = i * 0.1f;

        ace_buffer_t buf_in, buf_out;
        ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
        ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
        ace_buffer_write(buf_in, h_in, N * sizeof(float));

        int n = N;
        void* args[] = {&n, &alpha, buf_in, buf_out};
        int types[] = {ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF};

        ace_error_t err = ace_kernel_invoke(dev, _ace_get_scale(), ACE_DTYPE_FLOAT32, N, args, types, 4);
        ace_finish(dev);
        ace_buffer_read(buf_out, h_out, N * sizeof(float));

        int pass = 1;
        for (int i = 0; i < 10; i++) {
            float expected = h_in[i] * alpha;
            if (h_out[i] != expected) { pass = 0; break; }
        }
        printf("%s (err=%d)\n", pass ? "PASS" : "FAIL", err);
        printf("First 10: ");
        for (int i = 0; i < 10; i++) printf("%.2f ", h_out[i]);
        printf("\n\n");

        free(h_in); free(h_out);
        ace_buffer_free(buf_in);
        ace_buffer_free(buf_out);
    }

    ace_device_release(dev);
    printf("========================================\n");
    printf("  CUDA Tests Completed!\n");
    printf("========================================\n");
    return 0;
}
