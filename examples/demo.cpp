/**
 * @file demo.cpp
 * @brief AgierCompute kernel demo - generic kernel + auto registration
 */
#include <cstdio>
#include <cstring>
#include "ace.h"

/* ============================================================================
 * Generic kernel definitions (use T as type placeholder)
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

ACE_KERNEL(relu,
    void relu(int n, T* in, T* out) {
        int i = GID;
        if (i < n) out[i] = in[i] > 0 ? in[i] : 0;
    }
);

/* ============================================================================
 * Test functions
 * ============================================================================ */

void test_vec_add_float(ace_device_t dev) {
    printf("\n--- Test: vec_add (float) ---\n");
    
    const int N = 8;
    float h_a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_b[] = {10, 20, 30, 40, 50, 60, 70, 80};
    float h_c[8];
    
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
    
    bool pass = true;
    for (int i = 0; i < N; i++) if (h_c[i] != h_a[i] + h_b[i]) pass = false;
    printf("%s (err=%d)\n", pass ? "PASS" : "FAIL", err);
    printf("  Results: ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_c[i]);
    printf("\n");
    
    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
}

void test_vec_add_int(ace_device_t dev) {
    printf("\n--- Test: vec_add (int) ---\n");
    
    const int N = 6;
    int h_a[] = {100, 200, 300, 400, 500, 600};
    int h_b[] = {10, 20, 30, 40, 50, 60};
    int h_c[6];
    
    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, N * sizeof(int), &buf_a);
    ace_buffer_alloc(dev, N * sizeof(int), &buf_b);
    ace_buffer_alloc(dev, N * sizeof(int), &buf_c);
    
    ace_buffer_write(buf_a, h_a, N * sizeof(int));
    ace_buffer_write(buf_b, h_b, N * sizeof(int));
    
    int n = N;
    void* args[] = {&n, buf_a, buf_b, buf_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    
    ace_error_t err = ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_INT32, N, args, types, 4);
    ace_finish(dev);
    
    ace_buffer_read(buf_c, h_c, N * sizeof(int));
    
    bool pass = true;
    for (int i = 0; i < N; i++) if (h_c[i] != h_a[i] + h_b[i]) pass = false;
    printf("%s (err=%d)\n", pass ? "PASS" : "FAIL", err);
    printf("  Results: ");
    for (int i = 0; i < N; i++) printf("%d ", h_c[i]);
    printf("\n");
    
    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
}

void test_scale(ace_device_t dev) {
    printf("\n--- Test: scale (float) ---\n");
    
    const int N = 5;
    float h_in[] = {1, 2, 3, 4, 5};
    float h_out[5];
    float alpha = 2.5f;
    
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
    
    bool pass = true;
    for (int i = 0; i < N; i++) if (h_out[i] != h_in[i] * alpha) pass = false;
    printf("%s (err=%d)\n", pass ? "PASS" : "FAIL", err);
    printf("  Results: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_out[i]);
    printf("\n");
    
    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
}

void test_relu(ace_device_t dev) {
    printf("\n--- Test: relu (float) ---\n");
    
    const int N = 6;
    float h_in[] = {-3, -1, 0, 1, 2, 5};
    float h_out[6];
    
    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, h_in, N * sizeof(float));
    
    int n = N;
    void* args[] = {&n, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF};
    
    ace_error_t err = ace_kernel_invoke(dev, _ace_get_relu(), ACE_DTYPE_FLOAT32, N, args, types, 3);
    ace_finish(dev);
    
    ace_buffer_read(buf_out, h_out, N * sizeof(float));
    
    float expected[] = {0, 0, 0, 1, 2, 5};
    bool pass = true;
    for (int i = 0; i < N; i++) if (h_out[i] != expected[i]) pass = false;
    printf("%s (err=%d)\n", pass ? "PASS" : "FAIL", err);
    printf("  Results: ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_out[i]);
    printf("\n");
    
    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);
}

void test_backend(const char* name, ace_device_type_t type) {
    printf("\n=== Testing %s Backend ===\n", name);
    
    int count = 0;
    ace_error_t err = ace_device_count(type, &count);
    printf("%s devices: %d\n", name, count);
    
    if (count > 0) {
        ace_device_t dev = NULL;
        err = ace_device_get(type, 0, &dev);
        
        if (err == ACE_OK && dev) {
            test_vec_add_float(dev);
            test_vec_add_int(dev);
            test_scale(dev);
            test_relu(dev);
            ace_device_release(dev);
        } else {
            printf("Failed to get device (err=%d)\n", err);
        }
    } else {
        printf("No %s device available\n", name);
    }
}

int main() {
    printf("========================================\n");
    printf("  AgierCompute - Multi-Backend Demo\n");
    printf("========================================\n");
    fflush(stdout);
    
    /* Test available backends */
    test_backend("CPU", ACE_DEVICE_CPU);
    test_backend("OpenCL", ACE_DEVICE_OPENCL);
    test_backend("CUDA", ACE_DEVICE_CUDA);
    test_backend("Vulkan", ACE_DEVICE_VULKAN);
    
    printf("\n========================================\n");
    printf("  ALL TESTS COMPLETED!\n");
    printf("========================================\n");
    
    return 0;
}

