/**
 * @file benchmark.c
 * @brief AgierCompute 性能基准测试
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ace.h"
#include "lib/ace_test.h"

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/time.h>
#endif

/* 时间工具 */
static double get_time_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int freq_set = 0;
    if (!freq_set) { QueryPerformanceFrequency(&freq); freq_set = 1; }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
#endif
}

/* ============================================================================
 * 测试内核
 * ============================================================================ */

ACE_KERNEL(bench_vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
)

ACE_KERNEL(bench_vec_mul,
    void vec_mul(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] * b[i];
    }
)

/* ============================================================================
 * 配置
 * ============================================================================ */

typedef struct {
    size_t n;
    int iterations;
} benchmark_config_t;

/* ============================================================================
 * 向量加法测试
 * ============================================================================ */

static ace_test_result_t benchmark_vec_add(ace_device_t dev, void* user_data) {
    benchmark_config_t* cfg = (benchmark_config_t*)user_data;
    size_t N = cfg->n, bytes = N * sizeof(float);
    
    float *h_a = malloc(bytes), *h_b = malloc(bytes), *h_c = malloc(bytes);
    for (size_t i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }
    
    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, bytes, &buf_a);
    ace_buffer_alloc(dev, bytes, &buf_b);
    ace_buffer_alloc(dev, bytes, &buf_c);
    ace_buffer_write(buf_a, h_a, bytes);
    ace_buffer_write(buf_b, h_b, bytes);

    /* 预热 */
    ACE_INVOKE(dev, bench_vec_add, ACE_DTYPE_FLOAT32, N, N, buf_a, buf_b, buf_c);
    ace_finish(dev);

    /* 测试 */
    double start = get_time_ms();
    for (int i = 0; i < cfg->iterations; i++) {
        ACE_INVOKE(dev, bench_vec_add, ACE_DTYPE_FLOAT32, N, N, buf_a, buf_b, buf_c);
    }
    ace_finish(dev);
    double elapsed = (get_time_ms() - start) / cfg->iterations;
    
    /* 验证 */
    ace_buffer_read(buf_c, h_c, bytes);
    int ok = 1;
    for (size_t i = 0; i < 10 && ok; i++)
        if (fabs(h_c[i] - 3.0f) > 1e-5f) ok = 0;
    
    double bw = (3.0 * bytes / 1e9) / (elapsed / 1000.0);
    printf("  %.2f GB/s %s\n", bw, ok ? "OK" : "FAIL");
    
    free(h_a); free(h_b); free(h_c);
    ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);
    return ok ? ACE_TEST_PASS : ACE_TEST_FAIL;
}

/* ============================================================================
 * 向量乘法测试
 * ============================================================================ */

static ace_test_result_t benchmark_vec_mul(ace_device_t dev, void* user_data) {
    benchmark_config_t* cfg = (benchmark_config_t*)user_data;
    size_t N = cfg->n, bytes = N * sizeof(float);
    
    float *h_a = malloc(bytes), *h_b = malloc(bytes), *h_c = malloc(bytes);
    for (size_t i = 0; i < N; i++) { h_a[i] = 2.0f; h_b[i] = 3.0f; }
    
    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, bytes, &buf_a);
    ace_buffer_alloc(dev, bytes, &buf_b);
    ace_buffer_alloc(dev, bytes, &buf_c);
    ace_buffer_write(buf_a, h_a, bytes);
    ace_buffer_write(buf_b, h_b, bytes);

    ACE_INVOKE(dev, bench_vec_mul, ACE_DTYPE_FLOAT32, N, N, buf_a, buf_b, buf_c);
    ace_finish(dev);

    double start = get_time_ms();
    for (int i = 0; i < cfg->iterations; i++) {
        ACE_INVOKE(dev, bench_vec_mul, ACE_DTYPE_FLOAT32, N, N, buf_a, buf_b, buf_c);
    }
    ace_finish(dev);
    double elapsed = (get_time_ms() - start) / cfg->iterations;
    
    ace_buffer_read(buf_c, h_c, bytes);
    int ok = 1;
    for (size_t i = 0; i < 10 && ok; i++)
        if (fabs(h_c[i] - 6.0f) > 1e-5f) ok = 0;
    
    double bw = (3.0 * bytes / 1e9) / (elapsed / 1000.0);
    printf("  %.2f GB/s %s\n", bw, ok ? "OK" : "FAIL");
    
    free(h_a); free(h_b); free(h_c);
    ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);
    return ok ? ACE_TEST_PASS : ACE_TEST_FAIL;
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main() {
    printf("========================================\n  AgierCompute Benchmark\n========================================\n\n");
    
    benchmark_config_t cfg = { .n = 1024 * 1024, .iterations = 10 };
    printf("Config: %zu elements (%.1f MB), %d iterations\n\n", 
           cfg.n, cfg.n * sizeof(float) / 1e6, cfg.iterations);
    
    ace_test_case_t tests[] = {
        ACE_TEST_DEFINE("vec_add", benchmark_vec_add, &cfg),
        ACE_TEST_DEFINE("vec_mul", benchmark_vec_mul, &cfg),
    };
    
    ace_test_suite_t suite = { .name = "Benchmark", .tests = tests, .test_count = 2 };
    ace_test_suite_run(&suite);
    
    return 0;
}
