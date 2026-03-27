/**
 * @file test_dtype.c
 * @brief 数据类型测试 - 测试所有支持的数据类型
 *
 * 测试范围:
 * - 后端：CUDA, OpenCL, Vulkan
 * - 数据类型：FLOAT32, FLOAT64, INT32, INT64, FLOAT16, BFLOAT16, INT8, UINT8, INT16
 * - 操作：ADD, MUL, SUB, DIV, SCALE
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"

/* ============================================================================
 * 内核定义 - 使用通用内核函数
 * ============================================================================ */

ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = kadd(a[i], b[i]);
    }
)

ACE_KERNEL(vec_mul,
    void vec_mul(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = kmul(a[i], b[i]);
    }
)

ACE_KERNEL(vec_sub,
    void vec_sub(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = ksub(a[i], b[i]);
    }
)

ACE_KERNEL(vec_div,
    void vec_div(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = kdiv(a[i], b[i]);
    }
)

ACE_KERNEL(vec_scale,
    void vec_scale(int n, T alpha, T* a, T* c) {
        int i = GID;
        if (i < n) c[i] = kmul(alpha, a[i]);
    }
)

/* ============================================================================
 * 测试配置
 * ============================================================================ */

typedef enum {
    OP_ADD,
    OP_MUL,
    OP_SUB,
    OP_DIV,
    OP_SCALE,
    OP_COUNT
} op_type_t;

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
    {ACE_DTYPE_FLOAT32,  "FLOAT32",  sizeof(float),       1, 1, 1e-3,  0.5},
    {ACE_DTYPE_FLOAT64,  "FLOAT64",  sizeof(double),      1, 1, 1e-6,  0.5},
    {ACE_DTYPE_INT32,    "INT32",    sizeof(int32_t),     0, 1, 0,     1.0},
    {ACE_DTYPE_INT64,    "INT64",    sizeof(int64_t),     0, 1, 0,     1.0},
    {ACE_DTYPE_FLOAT16,  "FLOAT16",  2,                   1, 1, 0.1,   0.5},
    {ACE_DTYPE_BFLOAT16, "BFLOAT16", 2,                   1, 1, 0.1,   0.5},
    {ACE_DTYPE_INT8,     "INT8",     sizeof(int8_t),      0, 1, 0,     0.1},
    {ACE_DTYPE_UINT8,    "UINT8",    sizeof(uint8_t),     0, 0, 0,     0.1},
    {ACE_DTYPE_INT16,    "INT16",    sizeof(int16_t),     0, 1, 0,     0.5},
};

#define NUM_DTYPES (sizeof(dtype_configs) / sizeof(dtype_configs[0]))

static const char* op_names[] = {"ADD", "MUL", "SUB", "DIV", "SCALE"};

/* ============================================================================
 * 测试统计
 * ============================================================================ */

typedef struct {
    int passed;
    int failed;
    int skipped;
} test_stats_t;

static test_stats_t g_stats = {0, 0, 0};

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
    } else if (cfg->dtype == ACE_DTYPE_FLOAT64) {
        double* a = (double*)data_a;
        double* b = (double*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (i % 10) * scale * 0.1;
            b[i] = ((i * 2) % 10) * scale * 0.1;
        }
    } else if (cfg->dtype == ACE_DTYPE_FLOAT32) {
        float* a = (float*)data_a;
        float* b = (float*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (float)((i % 10) * scale * 0.1);
            b[i] = (float)(((i * 2) % 10) * scale * 0.1);
        }
    } else if (cfg->dtype == ACE_DTYPE_INT8) {
        int8_t* a = (int8_t*)data_a;
        int8_t* b = (int8_t*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (int8_t)((i % 10) - 5);
            b[i] = (int8_t)(((i * 2) % 10) - 5);
        }
    } else if (cfg->dtype == ACE_DTYPE_UINT8) {
        uint8_t* a = (uint8_t*)data_a;
        uint8_t* b = (uint8_t*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (uint8_t)(i % 10);
            b[i] = (uint8_t)((i * 2) % 10);
        }
    } else if (cfg->dtype == ACE_DTYPE_INT16) {
        int16_t* a = (int16_t*)data_a;
        int16_t* b = (int16_t*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (int16_t)((i % 10) - 5);
            b[i] = (int16_t)(((i * 2) % 10) - 5);
        }
    } else if (cfg->dtype == ACE_DTYPE_INT32) {
        int32_t* a = (int32_t*)data_a;
        int32_t* b = (int32_t*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (i % 10) - 5;
            b[i] = ((i * 2) % 10) - 5;
        }
    } else if (cfg->dtype == ACE_DTYPE_INT64) {
        int64_t* a = (int64_t*)data_a;
        int64_t* b = (int64_t*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (i % 10) - 5;
            b[i] = ((i * 2) % 10) - 5;
        }
    }
}

/* ============================================================================
 * 验证函数
 * ============================================================================ */

static double get_value(void* data, int idx, dtype_config_t* cfg) {
    if (cfg->dtype == ACE_DTYPE_FLOAT16) {
        return (double)float16_to_float(((uint16_t*)data)[idx]);
    } else if (cfg->dtype == ACE_DTYPE_BFLOAT16) {
        return (double)bfloat16_to_float(((uint16_t*)data)[idx]);
    } else if (cfg->dtype == ACE_DTYPE_FLOAT32) {
        return (double)((float*)data)[idx];
    } else if (cfg->dtype == ACE_DTYPE_FLOAT64) {
        return ((double*)data)[idx];
    } else if (cfg->dtype == ACE_DTYPE_INT8) {
        return (double)((int8_t*)data)[idx];
    } else if (cfg->dtype == ACE_DTYPE_UINT8) {
        return (double)((uint8_t*)data)[idx];
    } else if (cfg->dtype == ACE_DTYPE_INT16) {
        return (double)((int16_t*)data)[idx];
    } else if (cfg->dtype == ACE_DTYPE_INT32) {
        return (double)((int32_t*)data)[idx];
    } else if (cfg->dtype == ACE_DTYPE_INT64) {
        return (double)((int64_t*)data)[idx];
    }
    return 0.0;
}

static int verify_result(void* h_a, void* h_b, void* h_c, int idx, 
                         dtype_config_t* cfg, op_type_t op) {
    double a = get_value(h_a, idx, cfg);
    double b = get_value(h_b, idx, cfg);
    double c = get_value(h_c, idx, cfg);
    double expected = 0.0;

    switch (op) {
        case OP_ADD:   expected = a + b; break;
        case OP_MUL:   expected = a * b; break;
        case OP_SUB:   expected = a - b; break;
        case OP_DIV:
            if (fabs(b) < 1e-10) return 1; /* 跳过除零 */
            expected = a / b;
            break;
        case OP_SCALE: expected = 2.0 * a; break;
        default: return 0;
    }

    /* 处理无符号整数减法的溢出回绕 */
    if (!cfg->is_float && !cfg->is_signed && op == OP_SUB && expected < 0) {
        /* 对于无符号类型，负数结果会回绕 */
        double max_val = 0.0;
        if (cfg->dtype == ACE_DTYPE_UINT8) {
            max_val = 256.0;
        }
        if (max_val > 0) {
            expected = max_val + expected;  /* 回绕到正数 */
        }
    }

    double diff = fabs(c - expected);
    return (diff <= cfg->tolerance) ? 1 : 0;
}

/* ============================================================================
 * 测试执行
 * ============================================================================ */

static int test_op(ace_device_t dev, dtype_config_t* cfg, op_type_t op) {
    const int N = 64;
    size_t bytes = N * cfg->elem_size;

    void* h_a = malloc(bytes);
    void* h_b = malloc(bytes);
    void* h_c = malloc(bytes);
    memset(h_c, 0, bytes);

    init_test_data(h_a, h_b, N, cfg);

    ace_buffer_t buf_a, buf_b, buf_c;
    if (ace_buffer_alloc(dev, bytes, &buf_a) != ACE_OK) {
        free(h_a); free(h_b); free(h_c);
        return -1;
    }
    if (ace_buffer_alloc(dev, bytes, &buf_b) != ACE_OK) {
        ace_buffer_free(buf_a);
        free(h_a); free(h_b); free(h_c);
        return -1;
    }
    if (ace_buffer_alloc(dev, bytes, &buf_c) != ACE_OK) {
        ace_buffer_free(buf_a);
        ace_buffer_free(buf_b);
        free(h_a); free(h_b); free(h_c);
        return -1;
    }

    ace_buffer_write(buf_a, h_a, bytes);
    ace_buffer_write(buf_b, h_b, bytes);

    ace_error_t err = ACE_OK;
    int n = N;

    switch (op) {
        case OP_ADD:
            err = ace_kernel_invoke(dev, _ace_get_vec_add(), cfg->dtype, N,
                (void*[]){&n, buf_a, buf_b, buf_c},
                (int[]){sizeof(int), 0, 0, 0}, 4);
            break;
        case OP_MUL:
            err = ace_kernel_invoke(dev, _ace_get_vec_mul(), cfg->dtype, N,
                (void*[]){&n, buf_a, buf_b, buf_c},
                (int[]){sizeof(int), 0, 0, 0}, 4);
            break;
        case OP_SUB:
            err = ace_kernel_invoke(dev, _ace_get_vec_sub(), cfg->dtype, N,
                (void*[]){&n, buf_a, buf_b, buf_c},
                (int[]){sizeof(int), 0, 0, 0}, 4);
            break;
        case OP_DIV:
            err = ace_kernel_invoke(dev, _ace_get_vec_div(), cfg->dtype, N,
                (void*[]){&n, buf_a, buf_b, buf_c},
                (int[]){sizeof(int), 0, 0, 0}, 4);
            break;
        case OP_SCALE: {
            /* 根据数据类型设置 alpha */
            if (cfg->dtype == ACE_DTYPE_FLOAT64) {
                double alpha = 2.0;
                err = ace_kernel_invoke(dev, _ace_get_vec_scale(), cfg->dtype, N,
                    (void*[]){&n, &alpha, buf_a, buf_c},
                    (int[]){sizeof(int), sizeof(double), 0, 0}, 4);
            } else if (cfg->dtype == ACE_DTYPE_FLOAT16) {
                uint16_t alpha = float_to_float16(2.0f);
                err = ace_kernel_invoke(dev, _ace_get_vec_scale(), cfg->dtype, N,
                    (void*[]){&n, &alpha, buf_a, buf_c},
                    (int[]){sizeof(int), sizeof(uint16_t), 0, 0}, 4);
            } else if (cfg->dtype == ACE_DTYPE_BFLOAT16) {
                uint16_t alpha = float_to_bfloat16(2.0f);
                err = ace_kernel_invoke(dev, _ace_get_vec_scale(), cfg->dtype, N,
                    (void*[]){&n, &alpha, buf_a, buf_c},
                    (int[]){sizeof(int), sizeof(uint16_t), 0, 0}, 4);
            } else {
                float alpha = 2.0f;
                err = ace_kernel_invoke(dev, _ace_get_vec_scale(), cfg->dtype, N,
                    (void*[]){&n, &alpha, buf_a, buf_c},
                    (int[]){sizeof(int), sizeof(float), 0, 0}, 4);
            }
            break;
        }
        default:
            err = ACE_ERROR;
    }

    if (err != ACE_OK) {
        ace_buffer_free(buf_a);
        ace_buffer_free(buf_b);
        ace_buffer_free(buf_c);
        free(h_a); free(h_b); free(h_c);
        return 0;
    }

    ace_finish(dev);
    ace_buffer_read(buf_c, h_c, bytes);

    int ok = 1;
    for (int i = 0; i < 10 && ok; i++) {
        if (!verify_result(h_a, h_b, h_c, i, cfg, op)) {
            ok = 0;
        }
    }

    ace_buffer_free(buf_a);
    ace_buffer_free(buf_b);
    ace_buffer_free(buf_c);
    free(h_a); free(h_b); free(h_c);

    return ok ? 1 : 0;
}

/* ============================================================================
 * 设备测试
 * ============================================================================ */

static void test_device(ace_device_type_t type, int idx, const char* backend_name) {
    ace_device_t dev;
    if (ace_device_get(type, idx, &dev) != ACE_OK || !dev) return;

    ace_device_props_t props;
    ace_device_props(dev, &props);

    printf("\n----------------------------------------\n");
    printf(" Backend: %s\n", backend_name);
    printf(" Device:  %s\n", props.name);
    printf(" Memory:  %zu MB\n", props.total_memory / (1024 * 1024));
    printf("----------------------------------------\n\n");

    printf("Testing %d data types x %d operations:\n\n", (int)NUM_DTYPES, OP_COUNT);

    for (int d = 0; d < (int)NUM_DTYPES; d++) {
        dtype_config_t* cfg = &dtype_configs[d];

        for (int o = 0; o < OP_COUNT; o++) {
            /* 跳过整数类型的除法操作 */
            if (!cfg->is_float && (o == OP_DIV || o == OP_SCALE)) {
                printf("  [%2d/%2d][%2d/%2d] %-10s %-6s ... SKIP\n",
                       d + 1, (int)NUM_DTYPES, o + 1, OP_COUNT,
                       cfg->name, op_names[o]);
                g_stats.skipped++;
                continue;
            }

            printf("  [%2d/%2d][%2d/%2d] %-10s %-6s ... ",
                   d + 1, (int)NUM_DTYPES, o + 1, OP_COUNT,
                   cfg->name, op_names[o]);

            int result = test_op(dev, cfg, (op_type_t)o);
            if (result == 1) {
                printf("PASS\n");
                g_stats.passed++;
            } else if (result == 0) {
                printf("FAIL\n");
                g_stats.failed++;
            } else {
                printf("SKIP\n");
                g_stats.skipped++;
            }
        }
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
    printf("  Operations: ADD/MUL/SUB/DIV/SCALE\n");
    printf("========================================================\n");

    /* CUDA */
    int count = 0;
    ace_device_count(ACE_DEVICE_CUDA, &count);
    if (count > 0) {
        printf("\n>>> CUDA Devices: %d\n", count);
        for (int i = 0; i < count; i++)
            test_device(ACE_DEVICE_CUDA, i, "CUDA");
    }

    /* OpenCL */
    ace_device_count(ACE_DEVICE_OPENCL, &count);
    if (count > 0) {
        printf("\n>>> OpenCL Devices: %d\n", count);
        for (int i = 0; i < count; i++)
            test_device(ACE_DEVICE_OPENCL, i, "OpenCL");
    }

    /* Vulkan */
    ace_device_count(ACE_DEVICE_VULKAN, &count);
    if (count > 0) {
        printf("\n>>> Vulkan Devices: %d\n", count);
        for (int i = 0; i < count; i++)
            test_device(ACE_DEVICE_VULKAN, i, "Vulkan");
    }

    printf("\n========================================================\n");
    printf("  Results: %d passed, %d failed, %d skipped\n",
           g_stats.passed, g_stats.failed, g_stats.skipped);
    printf("========================================================\n");

    return (g_stats.failed > 0 || g_stats.passed == 0) ? 1 : 0;
}
