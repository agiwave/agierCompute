/**
 * @file test_dtype_debug.c
 * @brief 数据类型调试测试
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ace.h"

ACE_KERNEL(test_add,
    void test_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
)

int main() {
    printf("=== Data Type Debug Test ===\n\n");
    
    /* 测试 CUDA */
    printf("--- Testing CUDA ---\n");
    int cuda_count = 0;
    ace_device_count(ACE_DEVICE_CUDA, &cuda_count);
    printf("CUDA devices: %d\n", cuda_count);
    
    if (cuda_count > 0) {
        ace_device_t dev;
        ace_device_get(ACE_DEVICE_CUDA, 0, &dev);
        
        /* 测试 FLOAT16 */
        printf("\nTesting FLOAT16 on CUDA...\n");
        const int N = 10;
        size_t bytes = N * 2;  /* float16 is 2 bytes */
        
        uint16_t h_a[N], h_b[N], h_c[N];
        for (int i = 0; i < N; i++) {
            h_a[i] = 0x3C00;  /* 1.0 in float16 */
            h_b[i] = 0x4000;  /* 2.0 in float16 */
        }
        
        ace_buffer_t buf_a, buf_b, buf_c;
        ace_error_t err = ace_buffer_alloc(dev, bytes, &buf_a);
        printf("  alloc buf_a: %s\n", err == ACE_OK ? "OK" : "FAIL");
        
        err = ace_buffer_alloc(dev, bytes, &buf_b);
        printf("  alloc buf_b: %s\n", err == ACE_OK ? "OK" : "FAIL");
        
        err = ace_buffer_alloc(dev, bytes, &buf_c);
        printf("  alloc buf_c: %s\n", err == ACE_OK ? "OK" : "FAIL");
        
        err = ace_buffer_write(buf_a, h_a, bytes);
        printf("  write buf_a: %s\n", err == ACE_OK ? "OK" : "FAIL");
        
        err = ace_buffer_write(buf_b, h_b, bytes);
        printf("  write buf_b: %s\n", err == ACE_OK ? "OK" : "FAIL");
        
        /* 使用 ACE_INVOKE 宏 */
        ACE_INVOKE(dev, test_add, ACE_DTYPE_FLOAT16, N, N, buf_a, buf_b, buf_c);

        ace_finish(dev);
        err = ace_buffer_read(buf_c, h_c, bytes);
        printf("  read result: %s\n", err == ACE_OK ? "OK" : "FAIL");

        printf("  Results: ");
        for (int i = 0; i < N; i++) {
            printf("0x%04X ", h_c[i]);
        }
        printf("\n");

        ace_buffer_free(buf_a);
        ace_buffer_free(buf_b);
        ace_buffer_free(buf_c);
        ace_device_release(dev);
    }
    
    /* 测试 OpenCL */
    printf("\n--- Testing OpenCL ---\n");
    int opencl_count = 0;
    ace_device_count(ACE_DEVICE_OPENCL, &opencl_count);
    printf("OpenCL devices: %d\n", opencl_count);
    
    if (opencl_count > 0) {
        ace_device_t dev;
        ace_device_get(ACE_DEVICE_OPENCL, 0, &dev);
        
        /* 测试 FLOAT16 */
        printf("\nTesting FLOAT16 on OpenCL...\n");
        const int N = 10;
        size_t bytes = N * 2;
        
        uint16_t h_a[N], h_b[N], h_c[N];
        for (int i = 0; i < N; i++) {
            h_a[i] = 0x3C00;
            h_b[i] = 0x4000;
        }
        
        ace_buffer_t buf_a, buf_b, buf_c;
        ace_error_t err = ace_buffer_alloc(dev, bytes, &buf_a);
        printf("  alloc buf_a: %s\n", err == ACE_OK ? "OK" : "FAIL");
        
        err = ace_buffer_alloc(dev, bytes, &buf_b);
        printf("  alloc buf_b: %s\n", err == ACE_OK ? "OK" : "FAIL");
        
        err = ace_buffer_write(buf_a, h_a, bytes);
        printf("  write buf_a: %s\n", err == ACE_OK ? "OK" : "FAIL");
        
        err = ace_buffer_write(buf_b, h_b, bytes);
        printf("  write buf_b: %s\n", err == ACE_OK ? "OK" : "FAIL");

        /* 使用 ACE_INVOKE 宏 */
        ACE_INVOKE(dev, test_add, ACE_DTYPE_FLOAT16, N, N, buf_a, buf_b, buf_c);

        ace_device_release(dev);
    }
    
    printf("\n=== Test Complete ===\n");
    return 0;
}
