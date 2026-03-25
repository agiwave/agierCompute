/**
 * @file benchmark.c
 * @brief AgierCompute 性能基准测试
 *
 * 测试不同规模下的内核执行性能
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ace.h"

#ifdef _WIN32
    #include <windows.h>
    typedef LARGE_INTEGER ace_timer_t;
    static double g_freq = 0;
    static void timer_init(void) {
        QueryPerformanceFrequency(&g_freq);
    }
    static double timer_now(void) {
        ace_timer_t t;
        QueryPerformanceCounter(&t);
        return t.QuadPart / g_freq;
    }
#else
    #include <sys/time.h>
    typedef struct timeval ace_timer_t;
    static void timer_init(void) {}
    static double timer_now(void) {
        struct timeval t;
        gettimeofday(&t, NULL);
        return t.tv_sec + t.tv_usec / 1e6;
    }
#endif

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
 * 基准测试函数
 * ============================================================================ */

typedef struct {
    const char* name;
    double time_ms;
    double gflops;
    double bandwidth_gbs;
} benchmark_result_t;

static void benchmark_vec_add(ace_device_t dev, int N, benchmark_result_t* result) {
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_a);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_b);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_c);

    ace_buffer_write(buf_a, a, N * sizeof(float));
    ace_buffer_write(buf_b, b, N * sizeof(float));

    /* 预热 */
    int n = N;
    void* args[] = {&n, buf_a, buf_b, buf_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
    ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);

    /* 正式测试 */
    double start = timer_now();
    int iterations = 10;
    for (int i = 0; i < iterations; i++) {
        ace_kernel_invoke(dev, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    }
    ace_finish(dev);
    double end = timer_now();

    double time_ms = (end - start) * 1000 / iterations;
    double ops = 2.0 * N;  /* 一次加法 */
    double bytes = 3.0 * N * sizeof(float);  /* 读 2 写 1 */

    result->name = "vec_add";
    result->time_ms = time_ms;
    result->gflops = ops / time_ms / 1e6;
    result->bandwidth_gbs = bytes / time_ms / 1e6;

    ace_buffer_read(buf_c, c, N * sizeof(float));

    /* 验证 */
    int pass = 1;
    for (int i = 0; i < 10; i++) {
        if (c[i] != a[i] + b[i]) { pass = 0; break; }
    }

    free(a); free(b); free(c);
    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);

    printf("%-12s N=%8d: %8.3f ms, %8.3f GFLOPS, %8.3f GB/s [%s]\n",
           result->name, N, time_ms, result->gflops, result->bandwidth_gbs,
           pass ? "PASS" : "FAIL");
}

static void benchmark_scale(ace_device_t dev, int N, benchmark_result_t* result) {
    float *in = malloc(N * sizeof(float));
    float *out = malloc(N * sizeof(float));
    float alpha = 2.5f;

    for (int i = 0; i < N; i++) in[i] = i * 0.1f;

    ace_buffer_t buf_in, buf_out;
    ace_buffer_alloc(dev, N * sizeof(float), &buf_in);
    ace_buffer_alloc(dev, N * sizeof(float), &buf_out);
    ace_buffer_write(buf_in, in, N * sizeof(float));

    int n = N;
    void* args[] = {&n, &alpha, buf_in, buf_out};
    int types[] = {ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF};

    /* 预热 */
    ace_kernel_invoke(dev, _ace_get_scale(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish(dev);

    /* 正式测试 */
    double start = timer_now();
    int iterations = 10;
    for (int i = 0; i < iterations; i++) {
        ace_kernel_invoke(dev, _ace_get_scale(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    }
    ace_finish(dev);
    double end = timer_now();

    double time_ms = (end - start) * 1000 / iterations;
    double ops = 1.0 * N;  /* 一次乘法 */
    double bytes = 2.0 * N * sizeof(float);  /* 读 1 写 1 */

    result->name = "scale";
    result->time_ms = time_ms;
    result->gflops = ops / time_ms / 1e6;
    result->bandwidth_gbs = bytes / time_ms / 1e6;

    ace_buffer_read(buf_out, out, N * sizeof(float));

    free(in); free(out);
    ace_buffer_free(buf_in);
    ace_buffer_free(buf_out);

    printf("%-12s N=%8d: %8.3f ms, %8.3f GFLOPS, %8.3f GB/s\n",
           result->name, N, time_ms, result->gflops, result->bandwidth_gbs);
}

static void benchmark_gemm(ace_device_t dev, int M, int K, int N, benchmark_result_t* result) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *A = malloc(size_a);
    float *B = malloc(size_b);
    float *C = malloc(size_c);

    for (int i = 0; i < M * K; i++) A[i] = i % 10 * 0.1f;
    for (int i = 0; i < K * N; i++) B[i] = i % 7 * 0.1f;

    ace_buffer_t buf_A, buf_B, buf_C;
    ace_buffer_alloc(dev, size_a, &buf_A);
    ace_buffer_alloc(dev, size_b, &buf_B);
    ace_buffer_alloc(dev, size_c, &buf_C);

    ace_buffer_write(buf_A, A, size_a);
    ace_buffer_write(buf_B, B, size_b);

    void* args[] = {&M, &N, &K, buf_A, buf_B, buf_C};
    int types[] = {ACE_VAL, ACE_VAL, ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

    /* 预热 */
    ace_kernel_invoke(dev, _ace_get_gemm(), ACE_DTYPE_FLOAT32, M, args, types, 6);
    ace_finish(dev);

    /* 正式测试 */
    double start = timer_now();
    int iterations = 3;
    for (int i = 0; i < iterations; i++) {
        ace_kernel_invoke(dev, _ace_get_gemm(), ACE_DTYPE_FLOAT32, M, args, types, 6);
        ace_finish(dev);
    }
    double end = timer_now();

    double time_ms = (end - start) * 1000 / iterations;
    double ops = 2.0 * M * N * K;  /* 乘加 */
    double bytes = (M * K + K * N + M * N) * sizeof(float);

    result->name = "gemm";
    result->time_ms = time_ms;
    result->gflops = ops / time_ms / 1e6;
    result->bandwidth_gbs = bytes / time_ms / 1e6;

    ace_buffer_read(buf_C, C, size_c);

    free(A); free(B); free(C);
    ace_buffer_free(buf_A);
    ace_buffer_free(buf_B);
    ace_buffer_free(buf_C);

    printf("%-12s %dx%dx%d: %8.3f ms, %8.3f GFLOPS, %8.3f GB/s\n",
           result->name, M, K, N, time_ms, result->gflops, result->bandwidth_gbs);
}

static void benchmark_multi_device(ace_device_list_t* devices, int N, benchmark_result_t* result) {
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    ace_sharded_buffer_t sh_a, sh_b, sh_c;
    ace_buffer_alloc_sharded(devices, N * sizeof(float), &sh_a);
    ace_buffer_alloc_sharded(devices, N * sizeof(float), &sh_b);
    ace_buffer_alloc_sharded(devices, N * sizeof(float), &sh_c);

    ace_buffer_write_sharded(&sh_a, a, N * sizeof(float));
    ace_buffer_write_sharded(&sh_b, b, N * sizeof(float));

    int n = N;
    void* args[] = {&n, &sh_a, &sh_b, &sh_c};
    int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};

    /* 预热 */
    ace_kernel_invoke_sharded(devices, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
    ace_finish_all(devices);

    /* 正式测试 */
    double start = timer_now();
    int iterations = 10;
    for (int i = 0; i < iterations; i++) {
        ace_kernel_invoke_sharded(devices, _ace_get_vec_add(), ACE_DTYPE_FLOAT32, N, args, types, 4);
        ace_finish_all(devices);
    }
    double end = timer_now();

    double time_ms = (end - start) * 1000 / iterations;
    double ops = 2.0 * N;
    double bytes = 3.0 * N * sizeof(float);

    result->name = "vec_add_mp";
    result->time_ms = time_ms;
    result->gflops = ops / time_ms / 1e6;
    result->bandwidth_gbs = bytes / time_ms / 1e6;

    ace_buffer_read_sharded(&sh_c, c, N * sizeof(float));

    free(a); free(b); free(c);
    ace_buffer_free_sharded(&sh_a);
    ace_buffer_free_sharded(&sh_b);
    ace_buffer_free_sharded(&sh_c);

    printf("%-12s N=%8d: %8.3f ms, %8.3f GFLOPS, %8.3f GB/s [%d devices]\n",
           result->name, N, time_ms, result->gflops, result->bandwidth_gbs, devices->count);
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main() {
    printf("========================================\n");
    printf("  AgierCompute Benchmark\n");
    printf("========================================\n\n");

    timer_init();

    /* 获取设备 */
    ace_device_t dev;
    ace_error_t err = ace_device_select_best(&dev);
    if (err != ACE_OK || !dev) {
        printf("No device available\n");
        return 1;
    }

    printf("Using device:\n");
    ace_device_print_info(dev);
    printf("\n");

    /* 向量加法基准测试 - 不同规模 */
    printf("--- Vector Add Benchmark ---\n");
    int sizes[] = {100000, 500000, 1000000, 5000000, 10000000};
    benchmark_result_t result;

    for (int i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        benchmark_vec_add(dev, sizes[i], &result);
    }

    /* 缩放基准测试 */
    printf("\n--- Scale Benchmark ---\n");
    for (int i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        benchmark_scale(dev, sizes[i], &result);
    }

    /* 矩阵乘法基准测试 - 不同规模 */
    printf("\n--- GEMM Benchmark ---\n");
    int gemm_sizes[][3] = {
        {32, 32, 32},
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512}
    };

    for (int i = 0; i < 5; i++) {
        benchmark_gemm(dev, gemm_sizes[i][0], gemm_sizes[i][1], gemm_sizes[i][2], &result);
    }

    /* 多设备基准测试 */
    printf("\n--- Multi-Device Benchmark ---\n");
    ace_device_list_t devices;
    ace_device_get_all(&devices);

    if (devices.count > 1) {
        printf("Testing with %d devices:\n", devices.count);
        for (int i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
            benchmark_multi_device(&devices, sizes[i], &result);
        }
    } else {
        printf("Only 1 device available, skipping multi-device benchmark\n");
    }

    ace_device_list_release(&devices);
    ace_device_release(dev);

    printf("\n========================================\n");
    printf("  Benchmark Complete!\n");
    printf("========================================\n");

    return 0;
}
