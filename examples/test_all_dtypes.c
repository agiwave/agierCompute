/**
 * @file test_all_dtypes.c
 * @brief 完整数据类型测试 - 覆盖所有后端和所有数据类型
 *
 * 测试范围:
 * - 后端：CUDA, OpenCL, Vulkan
 * - 数据类型：FLOAT32, FLOAT64, INT32, INT64, FLOAT16, BFLOAT16, INT8, UINT8, INT16
 * - 操作：向量加法、向量乘法
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"

/* ============================================================================
 * 内核定义 - 统一使用泛型内核
 * 框架会自动处理不同数据类型的转换
 * ============================================================================ */

ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] + b[i];
    }
)

ACE_KERNEL(vec_mul,
    void vec_mul(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = a[i] * b[i];
    }
)

/* ============================================================================
 * 测试配置
 * ============================================================================ */

typedef struct {
    ace_dtype_t dtype;
    const char* name;
    size_t elem_size;
    int is_float;
    int is_signed;
    double tolerance;
    double test_scale;
} dtype_config_t;

static dtype_config_t dtype_configs[] = {
    /* 基础数据类型 */
    {ACE_DTYPE_FLOAT32,  "FLOAT32",  sizeof(float),       1, 1, 1e-3,  0.5},
    {ACE_DTYPE_FLOAT64,  "FLOAT64",  sizeof(double),      1, 1, 1e-6,  0.5},
    {ACE_DTYPE_INT32,    "INT32",    sizeof(int32_t),     0, 1, 0,     1.0},
    {ACE_DTYPE_INT64,    "INT64",    sizeof(int64_t),     0, 1, 0,     1.0},
    
    /* AI 重要数据类型 */
    {ACE_DTYPE_FLOAT16,  "FLOAT16",  2,                   1, 1, 0.1,   0.5},
    {ACE_DTYPE_BFLOAT16, "BFLOAT16", 2,                   1, 1, 0.1,   0.5},
    {ACE_DTYPE_INT8,     "INT8",     sizeof(int8_t),      0, 1, 0,     0.1},
    {ACE_DTYPE_UINT8,    "UINT8",    sizeof(uint8_t),     0, 0, 0,     0.1},
    {ACE_DTYPE_INT16,    "INT16",    sizeof(int16_t),     0, 1, 0,     0.5},
};

#define NUM_DTYPES (sizeof(dtype_configs) / sizeof(dtype_configs[0]))

/* ============================================================================
 * 辅助函数：初始化测试数据
 * ============================================================================ */

static void init_data_fp16(uint16_t* a, uint16_t* b, int N, double scale, int is_bf16) {
    for (int i = 0; i < N; i++) {
        float fa = (float)((i % 10) * scale * 0.1);
        float fb = (float)(((i * 2) % 10) * scale * 0.1);
        if (is_bf16) {
            a[i] = float_to_bfloat16(fa);
            b[i] = float_to_bfloat16(fb);
        } else {
            a[i] = float_to_float16(fa);
            b[i] = float_to_float16(fb);
        }
    }
}

static void init_test_data(void* data_a, void* data_b, int N, dtype_config_t* cfg) {
    double scale = cfg->test_scale;
    
    if (cfg->dtype == ACE_DTYPE_FLOAT16) {
        init_data_fp16((uint16_t*)data_a, (uint16_t*)data_b, N, scale, 0);
    } else if (cfg->dtype == ACE_DTYPE_BFLOAT16) {
        init_data_fp16((uint16_t*)data_a, (uint16_t*)data_b, N, scale, 1);
    } else if (cfg->dtype == ACE_DTYPE_FLOAT32) {
        float* a = (float*)data_a;
        float* b = (float*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (float)(i * scale * 0.1);
            b[i] = (float)((i * 2) * scale * 0.1);
        }
    } else if (cfg->dtype == ACE_DTYPE_FLOAT64) {
        double* a = (double*)data_a;
        double* b = (double*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (double)(i * scale * 0.1);
            b[i] = (double)((i * 2) * scale * 0.1);
        }
    } else if (cfg->dtype == ACE_DTYPE_INT8) {
        int8_t* a = (int8_t*)data_a;
        int8_t* b = (int8_t*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (int8_t)((i % 20) - 10);
            b[i] = (int8_t)(((i * 2) % 20) - 10);
        }
    } else if (cfg->dtype == ACE_DTYPE_UINT8) {
        uint8_t* a = (uint8_t*)data_a;
        uint8_t* b = (uint8_t*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (uint8_t)(i % 50);
            b[i] = (uint8_t)((i * 2) % 50);
        }
    } else if (cfg->dtype == ACE_DTYPE_INT16) {
        int16_t* a = (int16_t*)data_a;
        int16_t* b = (int16_t*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (int16_t)((i % 100) - 50);
            b[i] = (int16_t)(((i * 2) % 100) - 50);
        }
    } else if (cfg->dtype == ACE_DTYPE_INT32) {
        int32_t* a = (int32_t*)data_a;
        int32_t* b = (int32_t*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (int32_t)(i - N/2);
            b[i] = (int32_t)((i * 2) - N);
        }
    } else if (cfg->dtype == ACE_DTYPE_INT64) {
        int64_t* a = (int64_t*)data_a;
        int64_t* b = (int64_t*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (int64_t)(i - N/2);
            b[i] = (int64_t)((i * 2) - N);
        }
    }
}

/* ============================================================================
 * 验证函数
 * ============================================================================ */

static int verify_results(void* h_a, void* h_b, void* h_c, int N, dtype_config_t* cfg, const char* op) {
    int ok = 1;
    int check_count = (N < 10) ? N : 10;
    
    if (cfg->dtype == ACE_DTYPE_FLOAT16) {
        uint16_t* a = (uint16_t*)h_a;
        uint16_t* b = (uint16_t*)h_b;
        uint16_t* c = (uint16_t*)h_c;
        for (int i = 0; i < check_count && ok; i++) {
            double fa = (double)float16_to_float(a[i]);
            double fb = (double)float16_to_float(b[i]);
            double fc = (double)float16_to_float(c[i]);
            double expected = (strcmp(op, "add") == 0) ? fa + fb : fa * fb;
            if (fabs(fc - expected) > cfg->tolerance) ok = 0;
        }
    } else if (cfg->dtype == ACE_DTYPE_BFLOAT16) {
        uint16_t* a = (uint16_t*)h_a;
        uint16_t* b = (uint16_t*)h_b;
        uint16_t* c = (uint16_t*)h_c;
        for (int i = 0; i < check_count && ok; i++) {
            double fa = (double)bfloat16_to_float(a[i]);
            double fb = (double)bfloat16_to_float(b[i]);
            double fc = (double)bfloat16_to_float(c[i]);
            double expected = (strcmp(op, "add") == 0) ? fa + fb : fa * fb;
            double diff = fabs(fc - expected);
            if (diff > cfg->tolerance) {
                printf("[DEBUG] BF16 %s: a=%f b=%f c=%f expected=%f diff=%f\n", 
                       op, fa, fb, fc, expected, diff);
                ok = 0;
            }
        }
    } else if (cfg->dtype == ACE_DTYPE_FLOAT32) {
        float* a = (float*)h_a;
        float* b = (float*)h_b;
        float* c = (float*)h_c;
        for (int i = 0; i < check_count && ok; i++) {
            float expected = (strcmp(op, "add") == 0) ? a[i] + b[i] : a[i] * b[i];
            if (fabsf(c[i] - expected) > (float)cfg->tolerance) ok = 0;
        }
    } else if (cfg->dtype == ACE_DTYPE_FLOAT64) {
        double* a = (double*)h_a;
        double* b = (double*)h_b;
        double* c = (double*)h_c;
        for (int i = 0; i < check_count && ok; i++) {
            double expected = (strcmp(op, "add") == 0) ? a[i] + b[i] : a[i] * b[i];
            if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
        }
    } else if (cfg->elem_size == 1) {
        if (cfg->is_signed) {
            int8_t* a = (int8_t*)h_a;
            int8_t* b = (int8_t*)h_b;
            int8_t* c = (int8_t*)h_c;
            for (int i = 0; i < check_count && ok; i++) {
                int8_t expected = (strcmp(op, "add") == 0) ? a[i] + b[i] : a[i] * b[i];
                if (c[i] != expected) ok = 0;
            }
        } else {
            uint8_t* a = (uint8_t*)h_a;
            uint8_t* b = (uint8_t*)h_b;
            uint8_t* c = (uint8_t*)h_c;
            for (int i = 0; i < check_count && ok; i++) {
                uint8_t expected = (strcmp(op, "add") == 0) ? a[i] + b[i] : a[i] * b[i];
                if (c[i] != expected) ok = 0;
            }
        }
    } else if (cfg->elem_size == 2) {
        if (cfg->is_signed) {
            int16_t* a = (int16_t*)h_a;
            int16_t* b = (int16_t*)h_b;
            int16_t* c = (int16_t*)h_c;
            for (int i = 0; i < check_count && ok; i++) {
                int16_t expected = (strcmp(op, "add") == 0) ? a[i] + b[i] : a[i] * b[i];
                if (c[i] != expected) ok = 0;
            }
        } else {
            uint16_t* a = (uint16_t*)h_a;
            uint16_t* b = (uint16_t*)h_b;
            uint16_t* c = (uint16_t*)h_c;
            for (int i = 0; i < check_count && ok; i++) {
                uint16_t expected = (strcmp(op, "add") == 0) ? a[i] + b[i] : a[i] * b[i];
                if (c[i] != expected) ok = 0;
            }
        }
    } else if (cfg->elem_size == 4) {
        int32_t* a = (int32_t*)h_a;
        int32_t* b = (int32_t*)h_b;
        int32_t* c = (int32_t*)h_c;
        for (int i = 0; i < check_count && ok; i++) {
            int32_t expected = (strcmp(op, "add") == 0) ? a[i] + b[i] : a[i] * b[i];
            if (c[i] != expected) ok = 0;
        }
    } else {
        int64_t* a = (int64_t*)h_a;
        int64_t* b = (int64_t*)h_b;
        int64_t* c = (int64_t*)h_c;
        for (int i = 0; i < check_count && ok; i++) {
            int64_t expected = (strcmp(op, "add") == 0) ? a[i] + b[i] : a[i] * b[i];
            if (c[i] != expected) ok = 0;
        }
    }
    
    return ok;
}

/* ============================================================================
 * 测试执行函数
 * ============================================================================ */

typedef struct {
    int passed;
    int failed;
    int skipped;
} test_stats_t;

static int test_dtype_op(ace_device_t dev, dtype_config_t* cfg, const char* op) {
    const int N = 64;
    size_t bytes = N * cfg->elem_size;
    
    void* h_a = malloc(bytes);
    void* h_b = malloc(bytes);
    void* h_c = malloc(bytes);
    memset(h_c, 0, bytes);
    
    init_test_data(h_a, h_b, N, cfg);
    
    ace_buffer_t buf_a, buf_b, buf_c;
    ace_error_t err;
    
    err = ace_buffer_alloc(dev, bytes, &buf_a);
    if (err != ACE_OK) { free(h_a); free(h_b); free(h_c); return -1; }
    err = ace_buffer_alloc(dev, bytes, &buf_b);
    if (err != ACE_OK) { ace_buffer_free(buf_a); free(h_a); free(h_b); free(h_c); return -1; }
    err = ace_buffer_alloc(dev, bytes, &buf_c);
    if (err != ACE_OK) { ace_buffer_free(buf_a); ace_buffer_free(buf_b); free(h_a); free(h_b); free(h_c); return -1; }
    
    ace_buffer_write(buf_a, h_a, bytes);
    ace_buffer_write(buf_b, h_b, bytes);

    ace_kernel_t kernel = (strcmp(op, "add") == 0) ? _ace_get_vec_add() : _ace_get_vec_mul();

    err = ace_kernel_invoke(dev, kernel, cfg->dtype, N,
                            (void*[]){&N, buf_a, buf_b, buf_c},
                            (int[]){(int)sizeof(int), 0, 0, 0}, 4);
    
    if (err != ACE_OK) {
        ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);
        free(h_a); free(h_b); free(h_c);
        return 0;
    }
    
    ace_finish(dev);
    
    err = ace_buffer_read(buf_c, h_c, bytes);
    if (err != ACE_OK) {
        ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);
        free(h_a); free(h_b); free(h_c);
        return 0;
    }
    
    int ok = verify_results(h_a, h_b, h_c, N, cfg, op);
    
    ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);
    free(h_a); free(h_b); free(h_c);
    
    return ok ? 1 : 0;
}

static void test_device(ace_device_type_t type, int idx, const char* backend_name, test_stats_t* stats) {
    ace_device_t dev;
    ace_error_t err = ace_device_get(type, idx, &dev);
    if (err != ACE_OK || !dev) return;
    
    ace_device_props_t props;
    ace_device_props(dev, &props);
    
    printf("\n----------------------------------------\n");
    printf(" Backend: %s\n", backend_name);
    printf(" Device:  %s\n", props.name);
    printf(" Memory:  %zu MB\n", props.total_memory / (1024 * 1024));
    printf("----------------------------------------\n\n");
    
    printf("Testing %d data types (ADD + MUL):\n\n", (int)NUM_DTYPES);
    
    for (int i = 0; i < (int)NUM_DTYPES; i++) {
        dtype_config_t* cfg = &dtype_configs[i];
        
        printf("  [%2d/%2d] %-10s ADD ... ", i + 1, (int)NUM_DTYPES, cfg->name);
        int result_add = test_dtype_op(dev, cfg, "add");
        if (result_add == 1) { printf("PASS\n"); stats->passed++; }
        else if (result_add == 0) { printf("FAIL\n"); stats->failed++; }
        else { printf("SKIP\n"); stats->skipped++; }
        
        printf("  [%2d/%2d] %-10s MUL ... ", i + 1, (int)NUM_DTYPES, cfg->name);
        int result_mul = test_dtype_op(dev, cfg, "mul");
        if (result_mul == 1) { printf("PASS\n"); stats->passed++; }
        else if (result_mul == 0) { printf("FAIL\n"); stats->failed++; }
        else { printf("SKIP\n"); stats->skipped++; }
    }
    
    ace_device_release(dev);
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main() {
    printf("========================================================\n");
    printf("  AgierCompute - Data Type Test Suite\n");
    printf("  Backends: CUDA, OpenCL, Vulkan\n");
    printf("  Types: F32/F64/I32/I64/FP16/BF16/I8/U8/I16\n");
    printf("========================================================\n\n");
    
    test_stats_t stats = {0, 0, 0};
    
    /* CUDA */
    int cuda_count = 0;
    ace_device_count(ACE_DEVICE_CUDA, &cuda_count);
    if (cuda_count > 0) {
        printf(">>> CUDA Devices: %d\n", cuda_count);
        for (int i = 0; i < cuda_count; i++)
            test_device(ACE_DEVICE_CUDA, i, "CUDA", &stats);
    } else {
        printf(">>> CUDA: No devices\n");
    }
    
    /* OpenCL */
    int opencl_count = 0;
    ace_device_count(ACE_DEVICE_OPENCL, &opencl_count);
    if (opencl_count > 0) {
        printf("\n>>> OpenCL Devices: %d\n", opencl_count);
        for (int i = 0; i < opencl_count; i++)
            test_device(ACE_DEVICE_OPENCL, i, "OpenCL", &stats);
    } else {
        printf("\n>>> OpenCL: No devices\n");
    }
    
    /* Vulkan */
    int vulkan_count = 0;
    ace_device_count(ACE_DEVICE_VULKAN, &vulkan_count);
    if (vulkan_count > 0) {
        printf("\n>>> Vulkan Devices: %d\n", vulkan_count);
        for (int i = 0; i < vulkan_count; i++)
            test_device(ACE_DEVICE_VULKAN, i, "Vulkan", &stats);
    } else {
        printf("\n>>> Vulkan: No devices\n");
    }
    
    printf("\n========================================================\n");
    printf("  Results: %d passed, %d failed, %d skipped\n", 
           stats.passed, stats.failed, stats.skipped);
    printf("========================================================\n");
    
    return (stats.failed > 0 || stats.passed == 0) ? 1 : 0;
}
