/**
 * @file simple_test.c
 * @brief 动态加载完整测试
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

typedef int ace_error_t;
typedef void* ace_device_t;
typedef void* ace_buffer_t;
typedef void* ace_kernel_t;

/* 参数类型标记 */
#define ACE_ARG_BUFFER  0
#define ACE_ARG_VALUE   1

typedef int (*ace_init_fn)(const char*);
typedef void (*ace_shutdown_fn)(void);
typedef int (*ace_device_count_fn)(int type, int* count);
typedef int (*ace_device_get_fn)(int type, int idx, ace_device_t* dev);
typedef void (*ace_device_release_fn)(ace_device_t dev);
typedef int (*ace_buffer_alloc_fn)(ace_device_t dev, size_t size, ace_buffer_t* buf);
typedef void (*ace_buffer_free_fn)(ace_buffer_t buf);
typedef int (*ace_buffer_write_fn)(ace_buffer_t buf, const void* data, size_t size);
typedef int (*ace_buffer_read_fn)(ace_buffer_t buf, void* data, size_t size);
typedef int (*ace_kernel_compile_fn)(ace_device_t dev, const char* name, const char* src, ace_kernel_t* kernel);
typedef void (*ace_kernel_free_fn)(ace_kernel_t kernel);
typedef int (*ace_kernel_launch_fn)(ace_kernel_t kernel, void* cfg, void** args, size_t* sizes, int n);

int main() {
    printf("=== Dynamic Load Test ===\n");
    fflush(stdout);
    
    HMODULE h = LoadLibraryA("ace_core.dll");
    if (!h) {
        printf("Failed to load ace_core.dll\n");
        return 1;
    }
    
    ace_init_fn ace_init = (ace_init_fn)GetProcAddress(h, "ace_init");
    ace_device_count_fn ace_device_count = (ace_device_count_fn)GetProcAddress(h, "ace_device_count");
    ace_device_get_fn ace_device_get = (ace_device_get_fn)GetProcAddress(h, "ace_device_get");
    ace_buffer_alloc_fn ace_buffer_alloc = (ace_buffer_alloc_fn)GetProcAddress(h, "ace_buffer_alloc");
    ace_buffer_write_fn ace_buffer_write = (ace_buffer_write_fn)GetProcAddress(h, "ace_buffer_write");
    ace_buffer_read_fn ace_buffer_read = (ace_buffer_read_fn)GetProcAddress(h, "ace_buffer_read");
    ace_kernel_compile_fn ace_kernel_compile = (ace_kernel_compile_fn)GetProcAddress(h, "ace_kernel_compile");
    ace_kernel_launch_fn ace_kernel_launch = (ace_kernel_launch_fn)GetProcAddress(h, "ace_kernel_launch");
    ace_buffer_free_fn ace_buffer_free = (ace_buffer_free_fn)GetProcAddress(h, "ace_buffer_free");
    ace_kernel_free_fn ace_kernel_free = (ace_kernel_free_fn)GetProcAddress(h, "ace_kernel_free");
    ace_device_release_fn ace_device_release = (ace_device_release_fn)GetProcAddress(h, "ace_device_release");
    ace_shutdown_fn ace_shutdown = (ace_shutdown_fn)GetProcAddress(h, "ace_shutdown");
    
    printf("Functions loaded\n");
    fflush(stdout);
    
    printf("Calling ace_init...\n");
    fflush(stdout);
    int init_ret = ace_init(NULL);
    printf("ace_init returned: %d\n", init_ret);
    fflush(stdout);
    
    printf("Calling ace_device_count...\n");
    fflush(stdout);
    int count = 0;
    ace_device_count(0, &count);  /* 0 = ACE_DEVICE_CPU */
    printf("CPU devices: %d\n", count);
    fflush(stdout);
    
    if (count > 0) {
        printf("Calling ace_device_get...\n");
        fflush(stdout);
        ace_device_t dev;
        int get_ret = ace_device_get(0, 0, &dev);
        printf("Device obtained, ret=%d, dev=%p\n", get_ret, dev);
        fflush(stdout);
        
        /* Test vec_add kernel */
        printf("Compiling kernel...\n");
        fflush(stdout);
        const char* kernel_src = 
            "KERNEL(vec_add, int n, global float* a, global float* b, global float* c)"
            "{ int i = GID; if (i < n) c[i] = a[i] + b[i]; }";
        
        ace_kernel_t kernel;
        int compile_ret = ace_kernel_compile(dev, "vec_add", kernel_src, &kernel);
        printf("Kernel compiled, ret=%d\n", compile_ret);
        fflush(stdout);
        
        /* Allocate buffers */
        printf("Allocating buffers...\n");
        fflush(stdout);
        const int N = 4;
        ace_buffer_t buf_a, buf_b, buf_c;
        int alloc_ret = ace_buffer_alloc(dev, N * sizeof(float), &buf_a);
        printf("buf_a alloc ret=%d\n", alloc_ret);
        fflush(stdout);
        ace_buffer_alloc(dev, N * sizeof(float), &buf_b);
        printf("buf_b alloced\n");
        fflush(stdout);
        ace_buffer_alloc(dev, N * sizeof(float), &buf_c);
        printf("buf_c alloced\n");
        fflush(stdout);
        
        float h_a[] = {1, 2, 3, 4};
        float h_b[] = {10, 20, 30, 40};
        float h_c[4] = {0};
        
        printf("Writing data...\n");
        fflush(stdout);
        ace_buffer_write(buf_a, h_a, N * sizeof(float));
        printf("buf_a written\n");
        fflush(stdout);
        ace_buffer_write(buf_b, h_b, N * sizeof(float));
        printf("buf_b written\n");
        fflush(stdout);
        
        /* Launch */
        printf("Preparing launch config...\n");
        fflush(stdout);
        int n = N;
        struct { size_t grid[3]; size_t block[3]; size_t smem; } cfg = {{1,1,1}, {4,1,1}, 0};
        printf("cfg prepared\n");
        fflush(stdout);
        
        printf("Preparing args...\n");
        fflush(stdout);
        void* args[4];
        args[0] = &n;
        args[1] = buf_a;  /* 传递ace_buffer_t句柄 */
        args[2] = buf_b;
        args[3] = buf_c;
        size_t sizes[4];
        sizes[0] = sizeof(int);        /* 普通值 */
        sizes[1] = ACE_ARG_BUFFER;     /* 缓冲区句柄 */
        sizes[2] = ACE_ARG_BUFFER;
        sizes[3] = ACE_ARG_BUFFER;
        printf("args prepared\n");
        fflush(stdout);
        
        printf("Launching kernel...\n");
        fflush(stdout);
        int launch_ret = ace_kernel_launch(kernel, &cfg, args, sizes, 4);
        printf("Kernel launched, ret=%d\n", launch_ret);
        fflush(stdout);
        
        ace_buffer_read(buf_c, h_c, N * sizeof(float));
        
        printf("Results: ");
        for (int i = 0; i < N; i++) {
            printf("%.0f ", h_c[i]);
        }
        printf("\n");
        
        ace_buffer_free(buf_a);
        ace_buffer_free(buf_b);
        ace_buffer_free(buf_c);
        ace_kernel_free(kernel);
        ace_device_release(dev);
    }
    
    ace_shutdown();
    printf("Done!\n");
    FreeLibrary(h);
    return 0;
}
