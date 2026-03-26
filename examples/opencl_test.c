/**
 * @file opencl_test.c
 * @brief OpenCL 后端调试测试
 */
#include <stdio.h>
#include <stdlib.h>
#include "ace.h"

ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
)

int main() {
    printf("========================================\n");
    printf("  OpenCL Backend Debug Test\n");
    printf("========================================\n\n");

    int count = 0;
    ace_device_count(ACE_DEVICE_OPENCL, &count);
    printf("OpenCL devices: %d\n\n", count);

    if (count == 0) {
        printf("No OpenCL device\n");
        return 1;
    }

    ace_device_t dev;
    ace_device_get(ACE_DEVICE_OPENCL, 0, &dev);

    ace_device_props_t props;
    ace_device_props(dev, &props);
    printf("Device: %s\n", props.name);
    printf("  Vendor: %s\n", props.vendor);
    printf("  Compute units: %d\n", props.compute_units);
    printf("\n");

    const int N = 100;
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    ace_buffer_t buf_a, buf_b, buf_c;
    printf("Allocating buffers...\n");
    ace_error_t err = ace_buffer_alloc(dev, N * sizeof(float), &buf_a);
    printf("  buf_a: %s\n", err == ACE_OK ? "OK" : "FAIL");
    
    err = ace_buffer_alloc(dev, N * sizeof(float), &buf_b);
    printf("  buf_b: %s\n", err == ACE_OK ? "OK" : "FAIL");
    
    err = ace_buffer_alloc(dev, N * sizeof(float), &buf_c);
    printf("  buf_c: %s\n", err == ACE_OK ? "OK" : "FAIL");

    printf("\nWriting data...\n");
    err = ace_buffer_write(buf_a, h_a, N * sizeof(float));
    printf("  write a: %s\n", err == ACE_OK ? "OK" : "FAIL");
    
    err = ace_buffer_write(buf_b, h_b, N * sizeof(float));
    printf("  write b: %s\n", err == ACE_OK ? "OK" : "FAIL");

    printf("\nCompiling kernel...\n");

    printf("Launching kernel...\n");
    ACE_INVOKE(dev, vec_add, ACE_DTYPE_FLOAT32, N, &N, buf_a, buf_b, buf_c);

    printf("\nSyncing...\n");
    ace_finish(dev);

    printf("\nReading results...\n");
    err = ace_buffer_read(buf_c, h_c, N * sizeof(float));
    printf("  read: %s\n", err == ACE_OK ? "OK" : "FAIL");

    printf("\nResults: ");
    for (int i = 0; i < 10; i++) printf("%.0f ", h_c[i]);
    printf("\n");

    int pass = 1;
    for (int i = 0; i < 10; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) { pass = 0; break; }
    }
    printf("\n%s\n", pass ? "PASS" : "FAIL");

    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
    ace_device_release(dev);

    return 0;
}
