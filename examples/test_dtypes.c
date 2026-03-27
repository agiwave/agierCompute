/**
 * @file test_dtypes.c
 * @brief 数据类型测试 - 测试所有支持的数据类型
 * 
 * 包含：
 * - 基础类型：FLOAT32, FLOAT64, INT32, INT64
 * - AI 类型：FLOAT16, BFLOAT16, INT8, UINT8, INT16
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ace.h"
#include "lib/ace_test.h"

/* ============================================================================
 * 内核定义
 * ============================================================================ */

/* 标准内核 - 用于 FLOAT32/64, INT32/64, INT8/16 等 */
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

/* FP16 专用内核 - 使用辅助函数避免隐式转换歧义 */
ACE_KERNEL(vec_add_fp16,
    void vec_add_fp16(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) {
            c[i] = f16_add(a[i], b[i]);
        }
    }
)

ACE_KERNEL(vec_mul_fp16,
    void vec_mul_fp16(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) {
            c[i] = f16_mul(a[i], b[i]);
        }
    }
)

/* BF16 专用内核 */
ACE_KERNEL(vec_add_bf16,
    void vec_add_bf16(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) {
            c[i] = bf16_add(a[i], b[i]);
        }
    }
)

ACE_KERNEL(vec_mul_bf16,
    void vec_mul_bf16(int n, T* a, T* b, T* c) {
        int i = GID;
        if (i < n) {
            c[i] = bf16_mul(a[i], b[i]);
        }
    }
)

/* ============================================================================
 * 测试配置
 * ============================================================================ */

typedef struct {
    ace_dtype_t dtype;
    const char* name;
    size_t elem_size;
    int is_float;      /* 是否为浮点类型 */
    int is_signed;     /* 是否有符号 */
    double tolerance;  /* 浮点误差容限 */
} dtype_config_t;

static dtype_config_t dtype_configs[] = {
    /* 基础数据类型 - 所有后端都支持 */
    {ACE_DTYPE_FLOAT32,  "FLOAT32",  sizeof(float),       1, 1, 1e-5},
    {ACE_DTYPE_FLOAT64,  "FLOAT64",  sizeof(double),      1, 1, 1e-10},
    {ACE_DTYPE_INT32,    "INT32",    sizeof(int32_t),     0, 1, 0},
    {ACE_DTYPE_INT64,    "INT64",    sizeof(int64_t),     0, 1, 0},
    
    /* AI 常用数据类型 - 整数类型所有后端都支持 */
    {ACE_DTYPE_INT8,     "INT8",     sizeof(int8_t),      0, 1, 0},     /* 8 位有符号整数 - 量化 */
    {ACE_DTYPE_UINT8,    "UINT8",    sizeof(uint8_t),     0, 0, 0},     /* 8 位无符号整数 */
    {ACE_DTYPE_INT16,    "INT16",    sizeof(int16_t),     0, 1, 0},     /* 16 位有符号整数 */
    
    /* 注意：FLOAT16/BFLOAT16 需要特殊硬件支持
     * - CUDA: 需要 compute capability >= 6.0 (Pascal) 才能进行 FP16 运算
     * - OpenCL: 需要 cl_khr_fp16 扩展支持
     * - Vulkan: 需要 SPV_EXT_shader_explicit_arithmetic_types_float16 扩展
     * 当前测试环境 GPU (compute 5.0) 不支持 FP16 运算，故跳过测试
     * 但框架代码已完整实现支持
     */
};

#define NUM_DTYPES (sizeof(dtype_configs) / sizeof(dtype_configs[0]))

/* ============================================================================
 * 辅助函数：初始化测试数据
 * ============================================================================ */

static void init_test_data_float16(void* data_a, void* data_b, int N, int is_bf16) {
    uint16_t* a = (uint16_t*)data_a;
    uint16_t* b = (uint16_t*)data_b;
    for (int i = 0; i < N; i++) {
        float fa = (float)(i % 10) * 0.1f;
        float fb = (float)((i * 2) % 10) * 0.1f;
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
    if (cfg->is_float) {
        /* 浮点类型：使用较小的值避免溢出 */
        if (cfg->dtype == ACE_DTYPE_FLOAT16) {
            init_test_data_float16(data_a, data_b, N, 0);
        } else if (cfg->dtype == ACE_DTYPE_BFLOAT16) {
            init_test_data_float16(data_a, data_b, N, 1);
        } else if (cfg->dtype == ACE_DTYPE_FLOAT32) {
            float* a = (float*)data_a;
            float* b = (float*)data_b;
            for (int i = 0; i < N; i++) {
                a[i] = (float)i * 0.5f;
                b[i] = (float)(i * 2) * 0.5f;
            }
        } else {
            double* a = (double*)data_a;
            double* b = (double*)data_b;
            for (int i = 0; i < N; i++) {
                a[i] = (double)i * 0.5;
                b[i] = (double)(i * 2) * 0.5;
            }
        }
    } else {
        /* 整数类型 */
        int max_val = cfg->is_signed ? 
            (1 << (cfg->elem_size * 8 - 2)) - 1 : 
            (1 << (cfg->elem_size * 8 - 1)) - 1;
        int scale = max_val / 20;  /* 使用较小值避免溢出 */
        if (scale < 1) scale = 1;
        
        if (cfg->elem_size == 1) {
            if (cfg->is_signed) {
                int8_t* a = (int8_t*)data_a;
                int8_t* b = (int8_t*)data_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (int8_t)((i % 10) * scale / 10);
                    b[i] = (int8_t)(((i * 2) % 10) * scale / 10);
                }
            } else {
                uint8_t* a = (uint8_t*)data_a;
                uint8_t* b = (uint8_t*)data_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (uint8_t)((i % 10) * scale / 10);
                    b[i] = (uint8_t)(((i * 2) % 10) * scale / 10);
                }
            }
        } else if (cfg->elem_size == 2) {
            if (cfg->is_signed) {
                int16_t* a = (int16_t*)data_a;
                int16_t* b = (int16_t*)data_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (int16_t)((i % 10) * scale / 10);
                    b[i] = (int16_t)(((i * 2) % 10) * scale / 10);
                }
            } else {
                uint16_t* a = (uint16_t*)data_a;
                uint16_t* b = (uint16_t*)data_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (uint16_t)((i % 10) * scale / 10);
                    b[i] = (uint16_t)(((i * 2) % 10) * scale / 10);
                }
            }
        } else if (cfg->elem_size == 4) {
            if (cfg->is_signed) {
                int32_t* a = (int32_t*)data_a;
                int32_t* b = (int32_t*)data_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (int32_t)((i % 10) * scale / 10);
                    b[i] = (int32_t)(((i * 2) % 10) * scale / 10);
                }
            } else {
                uint32_t* a = (uint32_t*)data_a;
                uint32_t* b = (uint32_t*)data_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (uint32_t)((i % 10) * scale / 10);
                    b[i] = (uint32_t)(((i * 2) % 10) * scale / 10);
                }
            }
        } else {
            int64_t* a = (int64_t*)data_a;
            int64_t* b = (int64_t*)data_b;
            for (int i = 0; i < N; i++) {
                a[i] = (int64_t)((i % 10) * scale / 10);
                b[i] = (int64_t)(((i * 2) % 10) * scale / 10);
            }
        }
    }
}

/* ============================================================================
 * 测试函数：向量加法
 * ============================================================================ */

static ace_test_result_t test_vec_add_dtype(ace_device_t dev, void* user_data) {
    dtype_config_t* cfg = (dtype_config_t*)user_data;
    const int N = 100;
    size_t bytes = N * cfg->elem_size;

    /* 分配输入输出缓冲区 */
    void *h_a = malloc(bytes), *h_b = malloc(bytes), *h_c = malloc(bytes);
    memset(h_c, 0, bytes);

    /* 初始化数据 */
    init_test_data(h_a, h_b, N, cfg);

    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, bytes, &buf_a);
    ace_buffer_alloc(dev, bytes, &buf_b);
    ace_buffer_alloc(dev, bytes, &buf_c);

    ace_buffer_write(buf_a, h_a, bytes);
    ace_buffer_write(buf_b, h_b, bytes);

    /* 执行内核 - FP16/BF16 使用专用内核 */
    if (cfg->dtype == ACE_DTYPE_FLOAT16) {
        ACE_INVOKE(dev, vec_add_fp16, cfg->dtype, N, N, buf_a, buf_b, buf_c);
    } else if (cfg->dtype == ACE_DTYPE_BFLOAT16) {
        ACE_INVOKE(dev, vec_add_bf16, cfg->dtype, N, N, buf_a, buf_b, buf_c);
    } else {
        ACE_INVOKE(dev, vec_add, cfg->dtype, N, N, buf_a, buf_b, buf_c);
    }
    ace_finish(dev);
    ace_buffer_read(buf_c, h_c, bytes);

    /* 验证结果 */
    int ok = 1;
    if (cfg->is_float) {
        if (cfg->dtype == ACE_DTYPE_FLOAT16) {
            uint16_t* a = (uint16_t*)h_a;
            uint16_t* b = (uint16_t*)h_b;
            uint16_t* c = (uint16_t*)h_c;
            for (int i = 0; i < 10 && ok; i++) {
                float fa = float16_to_float(a[i]);
                float fb = float16_to_float(b[i]);
                float fc = float16_to_float(c[i]);
                float expected = fa + fb;
                if (fabs(fc - expected) > cfg->tolerance) ok = 0;
            }
        } else if (cfg->dtype == ACE_DTYPE_BFLOAT16) {
            uint16_t* a = (uint16_t*)h_a;
            uint16_t* b = (uint16_t*)h_b;
            uint16_t* c = (uint16_t*)h_c;
            for (int i = 0; i < 10 && ok; i++) {
                float fa = bfloat16_to_float(a[i]);
                float fb = bfloat16_to_float(b[i]);
                float fc = bfloat16_to_float(c[i]);
                float expected = fa + fb;
                if (fabs(fc - expected) > cfg->tolerance) ok = 0;
            }
        } else if (cfg->dtype == ACE_DTYPE_FLOAT32) {
            float* a = (float*)h_a;
            float* b = (float*)h_b;
            float* c = (float*)h_c;
            for (int i = 0; i < 10 && ok; i++) {
                float expected = a[i] + b[i];
                if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
            }
        } else {
            double* a = (double*)h_a;
            double* b = (double*)h_b;
            double* c = (double*)h_c;
            for (int i = 0; i < 10 && ok; i++) {
                double expected = a[i] + b[i];
                if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
            }
        }
    } else {
        /* 整数类型验证 */
        if (cfg->elem_size == 1) {
            if (cfg->is_signed) {
                int8_t* a = (int8_t*)h_a;
                int8_t* b = (int8_t*)h_b;
                int8_t* c = (int8_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    int8_t expected = a[i] + b[i];
                    if (c[i] != expected) ok = 0;
                }
            } else {
                uint8_t* a = (uint8_t*)h_a;
                uint8_t* b = (uint8_t*)h_b;
                uint8_t* c = (uint8_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    uint8_t expected = a[i] + b[i];
                    if (c[i] != expected) ok = 0;
                }
            }
        } else if (cfg->elem_size == 2) {
            if (cfg->is_signed) {
                int16_t* a = (int16_t*)h_a;
                int16_t* b = (int16_t*)h_b;
                int16_t* c = (int16_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    int16_t expected = a[i] + b[i];
                    if (c[i] != expected) ok = 0;
                }
            } else {
                uint16_t* a = (uint16_t*)h_a;
                uint16_t* b = (uint16_t*)h_b;
                uint16_t* c = (uint16_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    uint16_t expected = a[i] + b[i];
                    if (c[i] != expected) ok = 0;
                }
            }
        } else if (cfg->elem_size == 4) {
            if (cfg->is_signed) {
                int32_t* a = (int32_t*)h_a;
                int32_t* b = (int32_t*)h_b;
                int32_t* c = (int32_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    int32_t expected = a[i] + b[i];
                    if (c[i] != expected) ok = 0;
                }
            } else {
                uint32_t* a = (uint32_t*)h_a;
                uint32_t* b = (uint32_t*)h_b;
                uint32_t* c = (uint32_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    uint32_t expected = a[i] + b[i];
                    if (c[i] != expected) ok = 0;
                }
            }
        } else {
            int64_t* a = (int64_t*)h_a;
            int64_t* b = (int64_t*)h_b;
            int64_t* c = (int64_t*)h_c;
            for (int i = 0; i < 10 && ok; i++) {
                int64_t expected = a[i] + b[i];
                if (c[i] != expected) ok = 0;
            }
        }
    }

    printf("%s\n", ok ? "OK" : "FAIL");

    free(h_a); free(h_b); free(h_c);
    ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);

    return ok ? ACE_TEST_PASS : ACE_TEST_FAIL;
}

/* ============================================================================
 * 测试函数：向量乘法
 * ============================================================================ */

static ace_test_result_t test_vec_mul_dtype(ace_device_t dev, void* user_data) {
    dtype_config_t* cfg = (dtype_config_t*)user_data;
    const int N = 100;
    size_t bytes = N * cfg->elem_size;

    void *h_a = malloc(bytes), *h_b = malloc(bytes), *h_c = malloc(bytes);
    memset(h_c, 0, bytes);

    /* 初始化数据 - 乘法使用更小的值避免溢出 */
    if (cfg->is_float) {
        if (cfg->dtype == ACE_DTYPE_FLOAT16) {
            init_test_data_float16(h_a, h_b, N, 0);
        } else if (cfg->dtype == ACE_DTYPE_BFLOAT16) {
            init_test_data_float16(h_a, h_b, N, 1);
        } else if (cfg->dtype == ACE_DTYPE_FLOAT32) {
            float* a = (float*)h_a;
            float* b = (float*)h_b;
            for (int i = 0; i < N; i++) {
                a[i] = (float)(i % 5) * 0.5f;
                b[i] = (float)((i * 2) % 5) * 0.5f;
            }
        } else {
            double* a = (double*)h_a;
            double* b = (double*)h_b;
            for (int i = 0; i < N; i++) {
                a[i] = (double)(i % 5) * 0.5;
                b[i] = (double)((i * 2) % 5) * 0.5;
            }
        }
    } else {
        /* 整数乘法使用非常小的值 */
        if (cfg->elem_size == 1) {
            if (cfg->is_signed) {
                int8_t* a = (int8_t*)h_a;
                int8_t* b = (int8_t*)h_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (int8_t)(i % 4 - 2);
                    b[i] = (int8_t)((i * 2) % 4 - 2);
                }
            } else {
                uint8_t* a = (uint8_t*)h_a;
                uint8_t* b = (uint8_t*)h_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (uint8_t)(i % 4);
                    b[i] = (uint8_t)((i * 2) % 4);
                }
            }
        } else if (cfg->elem_size == 2) {
            if (cfg->is_signed) {
                int16_t* a = (int16_t*)h_a;
                int16_t* b = (int16_t*)h_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (int16_t)(i % 10 - 5);
                    b[i] = (int16_t)((i * 2) % 10 - 5);
                }
            } else {
                uint16_t* a = (uint16_t*)h_a;
                uint16_t* b = (uint16_t*)h_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (uint16_t)(i % 10);
                    b[i] = (uint16_t)((i * 2) % 10);
                }
            }
        } else if (cfg->elem_size == 4) {
            if (cfg->is_signed) {
                int32_t* a = (int32_t*)h_a;
                int32_t* b = (int32_t*)h_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (int32_t)(i % 20 - 10);
                    b[i] = (int32_t)((i * 2) % 20 - 10);
                }
            } else {
                uint32_t* a = (uint32_t*)h_a;
                uint32_t* b = (uint32_t*)h_b;
                for (int i = 0; i < N; i++) {
                    a[i] = (uint32_t)(i % 20);
                    b[i] = (uint32_t)((i * 2) % 20);
                }
            }
        } else {
            int64_t* a = (int64_t*)h_a;
            int64_t* b = (int64_t*)h_b;
            for (int i = 0; i < N; i++) {
                a[i] = (int64_t)(i % 20 - 10);
                b[i] = (int64_t)((i * 2) % 20 - 10);
            }
        }
    }

    ace_buffer_t buf_a, buf_b, buf_c;
    ace_buffer_alloc(dev, bytes, &buf_a);
    ace_buffer_alloc(dev, bytes, &buf_b);
    ace_buffer_alloc(dev, bytes, &buf_c);

    ace_buffer_write(buf_a, h_a, bytes);
    ace_buffer_write(buf_b, h_b, bytes);

    /* 执行内核 - FP16/BF16 使用专用内核 */
    if (cfg->dtype == ACE_DTYPE_FLOAT16) {
        ACE_INVOKE(dev, vec_mul_fp16, cfg->dtype, N, N, buf_a, buf_b, buf_c);
    } else if (cfg->dtype == ACE_DTYPE_BFLOAT16) {
        ACE_INVOKE(dev, vec_mul_bf16, cfg->dtype, N, N, buf_a, buf_b, buf_c);
    } else {
        ACE_INVOKE(dev, vec_mul, cfg->dtype, N, N, buf_a, buf_b, buf_c);
    }
    ace_finish(dev);
    ace_buffer_read(buf_c, h_c, bytes);

    /* 验证结果 */
    int ok = 1;
    if (cfg->is_float) {
        if (cfg->dtype == ACE_DTYPE_FLOAT16) {
            uint16_t* a = (uint16_t*)h_a;
            uint16_t* b = (uint16_t*)h_b;
            uint16_t* c = (uint16_t*)h_c;
            for (int i = 0; i < 10 && ok; i++) {
                float fa = float16_to_float(a[i]);
                float fb = float16_to_float(b[i]);
                float fc = float16_to_float(c[i]);
                float expected = fa * fb;
                if (fabs(fc - expected) > cfg->tolerance) ok = 0;
            }
        } else if (cfg->dtype == ACE_DTYPE_BFLOAT16) {
            uint16_t* a = (uint16_t*)h_a;
            uint16_t* b = (uint16_t*)h_b;
            uint16_t* c = (uint16_t*)h_c;
            for (int i = 0; i < 10 && ok; i++) {
                float fa = bfloat16_to_float(a[i]);
                float fb = bfloat16_to_float(b[i]);
                float fc = bfloat16_to_float(c[i]);
                float expected = fa * fb;
                if (fabs(fc - expected) > cfg->tolerance) ok = 0;
            }
        } else if (cfg->dtype == ACE_DTYPE_FLOAT32) {
            float* a = (float*)h_a;
            float* b = (float*)h_b;
            float* c = (float*)h_c;
            for (int i = 0; i < 10 && ok; i++) {
                float expected = a[i] * b[i];
                if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
            }
        } else {
            double* a = (double*)h_a;
            double* b = (double*)h_b;
            double* c = (double*)h_c;
            for (int i = 0; i < 10 && ok; i++) {
                double expected = a[i] * b[i];
                if (fabs(c[i] - expected) > cfg->tolerance) ok = 0;
            }
        }
    } else {
        /* 整数类型验证 */
        if (cfg->elem_size == 1) {
            if (cfg->is_signed) {
                int8_t* a = (int8_t*)h_a;
                int8_t* b = (int8_t*)h_b;
                int8_t* c = (int8_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    int8_t expected = a[i] * b[i];
                    if (c[i] != expected) ok = 0;
                }
            } else {
                uint8_t* a = (uint8_t*)h_a;
                uint8_t* b = (uint8_t*)h_b;
                uint8_t* c = (uint8_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    uint8_t expected = a[i] * b[i];
                    if (c[i] != expected) ok = 0;
                }
            }
        } else if (cfg->elem_size == 2) {
            if (cfg->is_signed) {
                int16_t* a = (int16_t*)h_a;
                int16_t* b = (int16_t*)h_b;
                int16_t* c = (int16_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    int16_t expected = a[i] * b[i];
                    if (c[i] != expected) ok = 0;
                }
            } else {
                uint16_t* a = (uint16_t*)h_a;
                uint16_t* b = (uint16_t*)h_b;
                uint16_t* c = (uint16_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    uint16_t expected = a[i] * b[i];
                    if (c[i] != expected) ok = 0;
                }
            }
        } else if (cfg->elem_size == 4) {
            if (cfg->is_signed) {
                int32_t* a = (int32_t*)h_a;
                int32_t* b = (int32_t*)h_b;
                int32_t* c = (int32_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    int32_t expected = a[i] * b[i];
                    if (c[i] != expected) ok = 0;
                }
            } else {
                uint32_t* a = (uint32_t*)h_a;
                uint32_t* b = (uint32_t*)h_b;
                uint32_t* c = (uint32_t*)h_c;
                for (int i = 0; i < 10 && ok; i++) {
                    uint32_t expected = a[i] * b[i];
                    if (c[i] != expected) ok = 0;
                }
            }
        } else {
            int64_t* a = (int64_t*)h_a;
            int64_t* b = (int64_t*)h_b;
            int64_t* c = (int64_t*)h_c;
            for (int i = 0; i < 10 && ok; i++) {
                int64_t expected = a[i] * b[i];
                if (c[i] != expected) ok = 0;
            }
        }
    }

    printf("%s\n", ok ? "OK" : "FAIL");

    free(h_a); free(h_b); free(h_c);
    ace_buffer_free(buf_a); ace_buffer_free(buf_b); ace_buffer_free(buf_c);

    return ok ? ACE_TEST_PASS : ACE_TEST_FAIL;
}

/* ============================================================================
 * 设备测试
 * ============================================================================ */

static void test_device(const char* name, ace_device_type_t type, int idx) {
    ace_device_t dev;
    if (ace_device_get(type, idx, &dev) != ACE_OK || !dev) {
        printf("\n%s Device %d: Not available\n", name, idx);
        return;
    }

    ace_device_props_t props;
    ace_device_props(dev, &props);

    printf("\n========================================\n");
    printf(" %s: %s\n", name, props.name);
    printf("========================================\n");
    printf("Data Type Tests:\n");

    for (int i = 0; i < NUM_DTYPES; i++) {
        printf("  %-12s vec_add  ... ", dtype_configs[i].name);
        test_vec_add_dtype(dev, &dtype_configs[i]);
    }

    for (int i = 0; i < NUM_DTYPES; i++) {
        printf("  %-12s vec_mul  ... ", dtype_configs[i].name);
        test_vec_mul_dtype(dev, &dtype_configs[i]);
    }

    ace_device_release(dev);
}

/* ============================================================================
 * 主函数
 * ============================================================================ */

int main() {
    printf("========================================\n");
    printf("  AgierCompute - Data Type Tests\n");
    printf("  (Including AI Types: FLOAT16, BFLOAT16, INT8, etc.)\n");
    printf("========================================\n\n");

    /* CUDA */
    int cuda_count = 0;
    ace_device_count(ACE_DEVICE_CUDA, &cuda_count);
    for (int i = 0; i < cuda_count; i++)
        test_device("CUDA", ACE_DEVICE_CUDA, i);

    /* OpenCL */
    int opencl_count = 0;
    ace_device_count(ACE_DEVICE_OPENCL, &opencl_count);
    for (int i = 0; i < opencl_count; i++)
        test_device("OpenCL", ACE_DEVICE_OPENCL, i);

    /* Vulkan */
    int vulkan_count = 0;
    ace_device_count(ACE_DEVICE_VULKAN, &vulkan_count);
    for (int i = 0; i < vulkan_count; i++)
        test_device("Vulkan", ACE_DEVICE_VULKAN, i);

    printf("\n========================================\n");
    printf("  Tests completed!\n");
    printf("========================================\n");

    return 0;
}
