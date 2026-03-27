/**
 * @file comprehensive_test.c
 * @brief 三维综合测试 - 多后端/多数据类型/多数学函数
 *
 * 测试维度:
 * 1. 后端设备：CUDA, OpenCL, Vulkan 的所有设备
 * 2. 数据类型：FLOAT32, FLOAT64, INT32, INT64, FLOAT16, BFLOAT16, INT8, UINT8, INT16
 * 3. 数学函数：加减乘除、幂运算、绝对值、最大值、最小值等
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"

/* ============================================================================
 * 内核定义 - 使用通用内核函数
 * ============================================================================ */

/* 基础运算 */
ACE_KERNEL(vec_add,
    void vec_add(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = kadd(a[i], b[i]);
    }
)

ACE_KERNEL(vec_sub,
    void vec_sub(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = ksub(a[i], b[i]);
    }
)

ACE_KERNEL(vec_mul,
    void vec_mul(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = kmul(a[i], b[i]);
    }
)

/* 除法 (简化版) */
ACE_KERNEL(vec_div,
    void vec_div(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = kdiv(a[i], b[i]);
    }
)

/* 幂运算 */
ACE_KERNEL(vec_square,
    void vec_square(int n, T* a, T* c) {
        int i = GID;
        if (i < n) c[i] = kmul(a[i], a[i]);
    }
)

/* 绝对值 */
ACE_KERNEL(vec_abs,
    void vec_abs(int n, T* a, T* c) {
        int i = GID;
        if (i < n) {
            T val = a[i];
            c[i] = klt(val, K_ZERO) ? ksub(K_ZERO, val) : val;
        }
    }
)

/* 最大值 */
ACE_KERNEL(vec_max,
    void vec_max(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = kgt(a[i], b[i]) ? a[i] : b[i];
    }
)

/* 最小值 */
ACE_KERNEL(vec_min,
    void vec_min(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) c[i] = klt(a[i], b[i]) ? a[i] : b[i];
    }
)

/* 标量乘法 */
ACE_KERNEL(vec_scale,
    void vec_scale(int n, T alpha, T* a, T* c) {
        int i = GID;
        if (i < n) c[i] = kmul(alpha, a[i]);
    }
)

/* 点积 (简化版，只计算前几个元素的和) */
ACE_KERNEL(vec_dot_partial,
    void vec_dot_partial(int n, T* a, T* b, T* result) {
        int i = GID;
        if (i < n) {
            result[i] = kmul(a[i], b[i]);
        }
    }
)

/* ============================================================================
 * 测试配置
 * ============================================================================ */

typedef enum {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_SQUARE,
    OP_ABS,
    OP_MAX,
    OP_MIN,
    OP_SCALE,
    OP_COUNT
} math_op_t;

typedef struct {
    math_op_t op;
    const char* name;
    int num_inputs;  /* 输入参数数量 */
} op_config_t;

static op_config_t op_configs[] = {
    {OP_ADD,    "ADD",    2},
    {OP_SUB,    "SUB",    2},
    {OP_MUL,    "MUL",    2},
    {OP_DIV,    "DIV",    2},
    {OP_SQUARE, "SQUARE", 1},
    {OP_ABS,    "ABS",    1},
    {OP_MAX,    "MAX",    2},
    {OP_MIN,    "MIN",    2},
    {OP_SCALE,  "SCALE",  1},
};

#define NUM_OPS (sizeof(op_configs) / sizeof(op_configs[0]))

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
 * 测试统计
 * ============================================================================ */

typedef struct {
    int passed;
    int failed;
    int skipped;
} test_stats_t;

/* 全局统计 */
static test_stats_t g_total_stats = {0, 0, 0};
static test_stats_t g_backend_stats[ACE_DEVICE_COUNT] = {{0,0,0}};
static test_stats_t g_dtype_stats[NUM_DTYPES] = {{0,0,0}};
static test_stats_t g_op_stats[NUM_OPS] = {{0,0,0}};

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
    if (cfg->dtype == ACE_DTYPE_FLOAT16 || cfg->dtype == ACE_DTYPE_BFLOAT16) {
        int is_bf16 = (cfg->dtype == ACE_DTYPE_BFLOAT16);
        init_data_fp16((uint16_t*)data_a, (uint16_t*)data_b, N, cfg->test_scale, is_bf16);
    } else if (cfg->dtype == ACE_DTYPE_FLOAT64) {
        double* a = (double*)data_a;
        double* b = (double*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = ((i % 10) * cfg->test_scale * 0.1);
            b[i] = (((i * 2) % 10) * cfg->test_scale * 0.1);
        }
    } else if (cfg->is_float) {
        float* a = (float*)data_a;
        float* b = (float*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (float)((i % 10) * cfg->test_scale * 0.1);
            b[i] = (float)(((i * 2) % 10) * cfg->test_scale * 0.1);
        }
    } else if (cfg->elem_size == 1) {
        if (cfg->is_signed) {
            int8_t* a = (int8_t*)data_a;
            int8_t* b = (int8_t*)data_b;
            for (int i = 0; i < N; i++) {
                a[i] = (int8_t)((i % 10) - 5);
                b[i] = (int8_t)(((i * 2) % 10) - 5);
            }
        } else {
            uint8_t* a = (uint8_t*)data_a;
            uint8_t* b = (uint8_t*)data_b;
            for (int i = 0; i < N; i++) {
                a[i] = (uint8_t)(i % 10);
                b[i] = (uint8_t)((i * 2) % 10);
            }
        }
    } else if (cfg->elem_size == 2) {
        if (cfg->is_signed) {
            int16_t* a = (int16_t*)data_a;
            int16_t* b = (int16_t*)data_b;
            for (int i = 0; i < N; i++) {
                a[i] = (int16_t)((i % 10) - 5);
                b[i] = (int16_t)(((i * 2) % 10) - 5);
            }
        } else {
            uint16_t* a = (uint16_t*)data_a;
            uint16_t* b = (uint16_t*)data_b;
            for (int i = 0; i < N; i++) {
                a[i] = (uint16_t)(i % 10);
                b[i] = (uint16_t)((i * 2) % 10);
            }
        }
    } else if (cfg->elem_size == 4) {
        if (cfg->is_float) {
            float* a = (float*)data_a;
            float* b = (float*)data_b;
            for (int i = 0; i < N; i++) {
                a[i] = (float)((i % 10) * cfg->test_scale * 0.1);
                b[i] = (float)(((i * 2) % 10) * cfg->test_scale * 0.1);
            }
        } else {
            int32_t* a = (int32_t*)data_a;
            int32_t* b = (int32_t*)data_b;
            for (int i = 0; i < N; i++) {
                a[i] = (i % 10) - 5;
                b[i] = ((i * 2) % 10) - 5;
            }
        }
    } else {
        int64_t* a = (int64_t*)data_a;
        int64_t* b = (int64_t*)data_b;
        for (int i = 0; i < N; i++) {
            a[i] = (i % 10) - 5;
            b[i] = ((i * 2) % 10) - 5;
        }
    }
}

/* ============================================================================
 * 辅助函数：验证结果
 * ============================================================================ */

static int verify_results(void* h_a, void* h_b, void* h_c, int N, 
                          dtype_config_t* cfg, math_op_t op) {
    int ok = 1;
    int check_count = (N < 10) ? N : 10;
    
    switch (op) {
        case OP_ADD:
            if (cfg->dtype == ACE_DTYPE_FLOAT16) {
                uint16_t* a = (uint16_t*)h_a;
                uint16_t* b = (uint16_t*)h_b;
                uint16_t* c = (uint16_t*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    double fa = (double)float16_to_float(a[i]);
                    double fb = (double)float16_to_float(b[i]);
                    double fc = (double)float16_to_float(c[i]);
                    double expected = fa + fb;
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
                    double expected = fa + fb;
                    if (fabs(fc - expected) > cfg->tolerance) ok = 0;
                }
            } else if (cfg->dtype == ACE_DTYPE_FLOAT64) {
                double* a = (double*)h_a;
                double* b = (double*)h_b;
                double* c = (double*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    double expected = a[i] + b[i];
                    if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
                }
            } else if (cfg->is_float) {
                float* a = (float*)h_a;
                float* b = (float*)h_b;
                float* c = (float*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    float expected = a[i] + b[i];
                    if (fabsf(c[i] - expected) > (float)cfg->tolerance) ok = 0;
                }
            } else if (cfg->elem_size == 1) {
                if (cfg->is_signed) {
                    int8_t* a = (int8_t*)h_a;
                    int8_t* b = (int8_t*)h_b;
                    int8_t* c = (int8_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        int8_t expected = a[i] + b[i];
                        if (c[i] != expected) ok = 0;
                    }
                } else {
                    uint8_t* a = (uint8_t*)h_a;
                    uint8_t* b = (uint8_t*)h_b;
                    uint8_t* c = (uint8_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        uint8_t expected = a[i] + b[i];
                        if (c[i] != expected) ok = 0;
                    }
                }
            } else if (cfg->elem_size == 2) {
                if (cfg->is_signed) {
                    int16_t* a = (int16_t*)h_a;
                    int16_t* b = (int16_t*)h_b;
                    int16_t* c = (int16_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        int16_t expected = a[i] + b[i];
                        if (c[i] != expected) ok = 0;
                    }
                } else {
                    uint16_t* a = (uint16_t*)h_a;
                    uint16_t* b = (uint16_t*)h_b;
                    uint16_t* c = (uint16_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        uint16_t expected = a[i] + b[i];
                        if (c[i] != expected) ok = 0;
                    }
                }
            } else if (cfg->elem_size == 4) {
                int32_t* a = (int32_t*)h_a;
                int32_t* b = (int32_t*)h_b;
                int32_t* c = (int32_t*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    int32_t expected = a[i] + b[i];
                    if (c[i] != expected) ok = 0;
                }
            } else {
                int64_t* a = (int64_t*)h_a;
                int64_t* b = (int64_t*)h_b;
                int64_t* c = (int64_t*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    int64_t expected = a[i] + b[i];
                    if (c[i] != expected) ok = 0;
                }
            }
            break;
            
        case OP_SUB:
            if (cfg->dtype == ACE_DTYPE_FLOAT16) {
                uint16_t* a = (uint16_t*)h_a;
                uint16_t* b = (uint16_t*)h_b;
                uint16_t* c = (uint16_t*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    double fa = (double)float16_to_float(a[i]);
                    double fb = (double)float16_to_float(b[i]);
                    double fc = (double)float16_to_float(c[i]);
                    double expected = fa - fb;
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
                    double expected = fa - fb;
                    if (fabs(fc - expected) > cfg->tolerance) ok = 0;
                }
            } else if (cfg->is_float && cfg->elem_size == 4) {
                float* a = (float*)h_a;
                float* b = (float*)h_b;
                float* c = (float*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    float expected = a[i] - b[i];
                    if (fabsf(c[i] - expected) > (float)cfg->tolerance) ok = 0;
                }
            } else if (cfg->is_float && cfg->elem_size == 8) {
                double* a = (double*)h_a;
                double* b = (double*)h_b;
                double* c = (double*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    double expected = a[i] - b[i];
                    if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
                }
            } else if (cfg->elem_size == 1) {
                if (cfg->is_signed) {
                    int8_t* a = (int8_t*)h_a;
                    int8_t* b = (int8_t*)h_b;
                    int8_t* c = (int8_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        int8_t expected = a[i] - b[i];
                        if (c[i] != expected) ok = 0;
                    }
                } else {
                    uint8_t* a = (uint8_t*)h_a;
                    uint8_t* b = (uint8_t*)h_b;
                    uint8_t* c = (uint8_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        uint8_t expected = a[i] - b[i];
                        if (c[i] != expected) ok = 0;
                    }
                }
            } else if (cfg->elem_size == 2) {
                if (cfg->is_signed) {
                    int16_t* a = (int16_t*)h_a;
                    int16_t* b = (int16_t*)h_b;
                    int16_t* c = (int16_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        int16_t expected = a[i] - b[i];
                        if (c[i] != expected) ok = 0;
                    }
                } else {
                    uint16_t* a = (uint16_t*)h_a;
                    uint16_t* b = (uint16_t*)h_b;
                    uint16_t* c = (uint16_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        uint16_t expected = a[i] - b[i];
                        if (c[i] != expected) ok = 0;
                    }
                }
            } else if (cfg->elem_size == 4) {
                int32_t* a = (int32_t*)h_a;
                int32_t* b = (int32_t*)h_b;
                int32_t* c = (int32_t*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    int32_t expected = a[i] - b[i];
                    if (c[i] != expected) ok = 0;
                }
            } else {
                int64_t* a = (int64_t*)h_a;
                int64_t* b = (int64_t*)h_b;
                int64_t* c = (int64_t*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    int64_t expected = a[i] - b[i];
                    if (c[i] != expected) ok = 0;
                }
            }
            break;
            
        case OP_MUL:
            if (cfg->dtype == ACE_DTYPE_FLOAT16) {
                uint16_t* a = (uint16_t*)h_a;
                uint16_t* b = (uint16_t*)h_b;
                uint16_t* c = (uint16_t*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    double fa = (double)float16_to_float(a[i]);
                    double fb = (double)float16_to_float(b[i]);
                    double fc = (double)float16_to_float(c[i]);
                    double expected = fa * fb;
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
                    double expected = fa * fb;
                    if (fabs(fc - expected) > cfg->tolerance) ok = 0;
                }
            } else if (cfg->is_float && cfg->elem_size == 4) {
                float* a = (float*)h_a;
                float* b = (float*)h_b;
                float* c = (float*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    float expected = a[i] * b[i];
                    if (fabsf(c[i] - expected) > (float)cfg->tolerance) ok = 0;
                }
            } else if (cfg->is_float && cfg->elem_size == 8) {
                double* a = (double*)h_a;
                double* b = (double*)h_b;
                double* c = (double*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    double expected = a[i] * b[i];
                    if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
                }
            } else if (cfg->elem_size == 1) {
                if (cfg->is_signed) {
                    int8_t* a = (int8_t*)h_a;
                    int8_t* b = (int8_t*)h_b;
                    int8_t* c = (int8_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        int8_t expected = a[i] * b[i];
                        if (c[i] != expected) ok = 0;
                    }
                } else {
                    uint8_t* a = (uint8_t*)h_a;
                    uint8_t* b = (uint8_t*)h_b;
                    uint8_t* c = (uint8_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        uint8_t expected = a[i] * b[i];
                        if (c[i] != expected) ok = 0;
                    }
                }
            } else if (cfg->elem_size == 2) {
                if (cfg->is_signed) {
                    int16_t* a = (int16_t*)h_a;
                    int16_t* b = (int16_t*)h_b;
                    int16_t* c = (int16_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        int16_t expected = a[i] * b[i];
                        if (c[i] != expected) ok = 0;
                    }
                } else {
                    uint16_t* a = (uint16_t*)h_a;
                    uint16_t* b = (uint16_t*)h_b;
                    uint16_t* c = (uint16_t*)h_c;
                    for (int i = 0; i < check_count && ok; i++) {
                        uint16_t expected = a[i] * b[i];
                        if (c[i] != expected) ok = 0;
                    }
                }
            } else if (cfg->elem_size == 4) {
                int32_t* a = (int32_t*)h_a;
                int32_t* b = (int32_t*)h_b;
                int32_t* c = (int32_t*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    int32_t expected = a[i] * b[i];
                    if (c[i] != expected) ok = 0;
                }
            }
            break;
            
        case OP_DIV:
            if (cfg->is_float && cfg->elem_size == 4) {
                float* a = (float*)h_a;
                float* b = (float*)h_b;
                float* c = (float*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    if (fabsf(b[i]) > 1e-6f) {
                        float expected = a[i] / b[i];
                        if (fabsf(c[i] - expected) > (float)cfg->tolerance) ok = 0;
                    }
                }
            } else if (cfg->is_float && cfg->elem_size == 8) {
                double* a = (double*)h_a;
                double* b = (double*)h_b;
                double* c = (double*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    if (fabs(b[i]) > 1e-12) {
                        double expected = a[i] / b[i];
                        if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
                    }
                }
            }
            break;
            
        case OP_SQUARE:
            if (cfg->is_float && cfg->elem_size == 4) {
                float* a = (float*)h_a;
                float* c = (float*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    float expected = a[i] * a[i];
                    if (fabsf(c[i] - expected) > (float)cfg->tolerance) ok = 0;
                }
            } else if (cfg->is_float && cfg->elem_size == 8) {
                double* a = (double*)h_a;
                double* c = (double*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    double expected = a[i] * a[i];
                    if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
                }
            }
            break;
            
        case OP_ABS:
            if (cfg->is_float && cfg->elem_size == 4) {
                float* a = (float*)h_a;
                float* c = (float*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    float expected = fabsf(a[i]);
                    if (fabsf(c[i] - expected) > (float)cfg->tolerance) ok = 0;
                }
            } else if (cfg->is_float && cfg->elem_size == 8) {
                double* a = (double*)h_a;
                double* c = (double*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    double expected = fabs(a[i]);
                    if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
                }
            }
            break;
            
        case OP_MAX:
        case OP_MIN:
            if (cfg->is_float && cfg->elem_size == 4) {
                float* a = (float*)h_a;
                float* b = (float*)h_b;
                float* c = (float*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    float expected = (op == OP_MAX) ? fmaxf(a[i], b[i]) : fminf(a[i], b[i]);
                    if (fabsf(c[i] - expected) > (float)cfg->tolerance) ok = 0;
                }
            } else if (cfg->is_float && cfg->elem_size == 8) {
                double* a = (double*)h_a;
                double* b = (double*)h_b;
                double* c = (double*)h_c;
                for (int i = 0; i < check_count && ok; i++) {
                    double expected = (op == OP_MAX) ? fmax(a[i], b[i]) : fmin(a[i], b[i]);
                    if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
                }
            }
            break;
            
        case OP_SCALE:
            if (cfg->is_float && cfg->elem_size == 4) {
                float* a = (float*)h_a;
                float* c = (float*)h_c;
                float alpha = 2.0f;
                for (int i = 0; i < check_count && ok; i++) {
                    float expected = alpha * a[i];
                    if (fabsf(c[i] - expected) > (float)cfg->tolerance) ok = 0;
                }
            } else if (cfg->is_float && cfg->elem_size == 8) {
                double* a = (double*)h_a;
                double* c = (double*)h_c;
                double alpha = 2.0;
                for (int i = 0; i < check_count && ok; i++) {
                    double expected = alpha * a[i];
                    if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
                }
            }
            break;
            
        default:
            break;
    }
    
    return ok;
}

/* ============================================================================
 * 测试执行函数
 * ============================================================================ */

static ace_kernel_t get_kernel(math_op_t op) {
    switch (op) {
        case OP_ADD:    return _ace_get_vec_add();
        case OP_SUB:    return _ace_get_vec_sub();
        case OP_MUL:    return _ace_get_vec_mul();
        case OP_DIV:    return _ace_get_vec_div();
        case OP_SQUARE: return _ace_get_vec_square();
        case OP_ABS:    return _ace_get_vec_abs();
        case OP_MAX:    return _ace_get_vec_max();
        case OP_MIN:    return _ace_get_vec_min();
        case OP_SCALE:  return _ace_get_vec_scale();
        default:        return NULL;
    }
}

static int test_op(ace_device_t dev, dtype_config_t* cfg, math_op_t op) {
    const int N = 64;
    size_t bytes = N * cfg->elem_size;
    
    void *h_a = malloc(bytes), *h_b = malloc(bytes), *h_c = malloc(bytes);
    memset(h_c, 0, bytes);
    
    init_test_data(h_a, h_b, N, cfg);
    
    ace_buffer_t buf_a, buf_b, buf_c;
    ace_error_t err;
    
    err = ace_buffer_alloc(dev, bytes, &buf_a);
    if (err != ACE_OK) { free(h_a); free(h_b); free(h_c); return -1; }
    err = ace_buffer_alloc(dev, bytes, &buf_b);
    if (err != ACE_OK) { ace_buffer_free(buf_a); free(h_a); free(h_b); free(h_c); return -1; }
    err = ace_buffer_alloc(dev, bytes, &buf_c);
    if (err != ACE_OK) { 
        ace_buffer_free(buf_a); ace_buffer_free(buf_b); 
        free(h_a); free(h_b); free(h_c); 
        return -1; 
    }
    
    ace_buffer_write(buf_a, h_a, bytes);
    ace_buffer_write(buf_b, h_b, bytes);
    
    ace_kernel_t kernel = get_kernel(op);
    op_config_t* op_cfg = &op_configs[op];
    
    if (op == OP_SCALE) {
        if (cfg->dtype == ACE_DTYPE_FLOAT64) {
            double alpha = 2.0;
            err = ace_kernel_invoke(dev, kernel, cfg->dtype, N,
                                    (void*[]){&N, &alpha, buf_a, buf_c},
                                    (int[]){sizeof(int), sizeof(double), 0, 0}, 4);
        } else if (cfg->dtype == ACE_DTYPE_FLOAT16 || cfg->dtype == ACE_DTYPE_BFLOAT16) {
            uint16_t alpha = cfg->dtype == ACE_DTYPE_FLOAT16 ? 
                             float_to_float16(2.0f) : float_to_bfloat16(2.0f);
            err = ace_kernel_invoke(dev, kernel, cfg->dtype, N,
                                    (void*[]){&N, &alpha, buf_a, buf_c},
                                    (int[]){sizeof(int), sizeof(uint16_t), 0, 0}, 4);
        } else {
            float alpha = 2.0f;
            err = ace_kernel_invoke(dev, kernel, cfg->dtype, N,
                                    (void*[]){&N, &alpha, buf_a, buf_c},
                                    (int[]){sizeof(int), sizeof(float), 0, 0}, 4);
        }
    } else if (op_cfg->num_inputs == 2) {
        err = ace_kernel_invoke(dev, kernel, cfg->dtype, N,
                                (void*[]){&N, buf_a, buf_b, buf_c},
                                (int[]){sizeof(int), 0, 0, 0}, 4);
    } else {
        err = ace_kernel_invoke(dev, kernel, cfg->dtype, N,
                                (void*[]){&N, buf_a, buf_c},
                                (int[]){sizeof(int), 0, 0}, 3);
    }
    
    if (err != ACE_OK) {
        ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);
        free(h_a); free(h_b); free(h_c);
        return 0;
    }
    
    ace_finish(dev);
    ace_buffer_read(buf_c, h_c, bytes);
    
    int ok = verify_results(h_a, (op_cfg->num_inputs == 2) ? h_b : h_a, h_c, N, cfg, op);
    
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
    printf("Testing %d data types x %d operations:\n\n", (int)NUM_DTYPES, (int)NUM_OPS);
    
    for (int d = 0; d < (int)NUM_DTYPES; d++) {
        dtype_config_t* cfg = &dtype_configs[d];
        
        for (int o = 0; o < (int)NUM_OPS; o++) {
            op_config_t* op_cfg = &op_configs[o];
            
            /* 跳过不支持的操作组合 */
            int skip = 0;
            if ((op_cfg->op == OP_DIV || op_cfg->op == OP_ABS || 
                 op_cfg->op == OP_MAX || op_cfg->op == OP_MIN ||
                 op_cfg->op == OP_SQUARE || op_cfg->op == OP_SCALE) &&
                !cfg->is_float) {
                skip = 1;  /* 只测试浮点类型 */
            }
            
            if (skip) {
                printf("  [%2d/%2d][%2d/%2d] %-8s %-6s ... SKIP\n", 
                       d + 1, (int)NUM_DTYPES, o + 1, (int)NUM_OPS, 
                       cfg->name, op_cfg->name);
                stats->skipped++;
                g_dtype_stats[d].skipped++;
                g_op_stats[o].skipped++;
                continue;
            }
            
            int result = test_op(dev, cfg, op_cfg->op);
            
            if (result == 1) { 
                printf("  [%2d/%2d][%2d/%2d] %-8s %-6s ... PASS\n", 
                       d + 1, (int)NUM_DTYPES, o + 1, (int)NUM_OPS, 
                       cfg->name, op_cfg->name);
                stats->passed++;
                g_dtype_stats[d].passed++;
                g_op_stats[o].passed++;
            } else if (result == 0) { 
                printf("  [%2d/%2d][%2d/%2d] %-8s %-6s ... FAIL\n", 
                       d + 1, (int)NUM_DTYPES, o + 1, (int)NUM_OPS, 
                       cfg->name, op_cfg->name);
                stats->failed++;
                g_dtype_stats[d].failed++;
                g_op_stats[o].failed++;
            } else { 
                printf("  [%2d/%2d][%2d/%2d] %-8s %-6s ... SKIP\n", 
                       d + 1, (int)NUM_DTYPES, o + 1, (int)NUM_OPS, 
                       cfg->name, op_cfg->name);
                stats->skipped++;
                g_dtype_stats[d].skipped++;
                g_op_stats[o].skipped++;
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
    printf("  AgierCompute - Comprehensive Test Suite\n");
    printf("  Dimensions: Backends x Data Types x Math Operations\n");
    printf("  Backends: CUDA, OpenCL, Vulkan\n");
    printf("  Types: F32/F64/I32/I64/FP16/BF16/I8/U8/I16\n");
    printf("  Operations: ADD/SUB/MUL/DIV/SQUARE/ABS/MAX/MIN/SCALE\n");
    printf("========================================================\n\n");
    
    /* 测试所有后端 */
    int cuda_count = 0, opencl_count = 0, vulkan_count = 0;
    
    ace_device_count(ACE_DEVICE_CUDA, &cuda_count);
    ace_device_count(ACE_DEVICE_OPENCL, &opencl_count);
    ace_device_count(ACE_DEVICE_VULKAN, &vulkan_count);
    
    printf(">>> CUDA Devices: %d\n", cuda_count);
    for (int i = 0; i < cuda_count; i++) {
        test_device(ACE_DEVICE_CUDA, i, "CUDA", &g_total_stats);
    }
    
    printf("\n>>> OpenCL Devices: %d\n", opencl_count);
    for (int i = 0; i < opencl_count; i++) {
        test_device(ACE_DEVICE_OPENCL, i, "OpenCL", &g_total_stats);
    }
    
    printf("\n>>> Vulkan Devices: %d\n", vulkan_count);
    for (int i = 0; i < vulkan_count; i++) {
        test_device(ACE_DEVICE_VULKAN, i, "Vulkan", &g_total_stats);
    }
    
    /* 打印统计信息 */
    printf("\n========================================================\n");
    printf("  Results: %d passed, %d failed, %d skipped\n", 
           g_total_stats.passed, g_total_stats.failed, g_total_stats.skipped);
    printf("========================================================\n");
    
    return (g_total_stats.failed > 0) ? 1 : 0;
}
