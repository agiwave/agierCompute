/**
 * @file cuda_type_utils.c
 * @brief CUDA type translation utilities
 */
#include "cuda_backend.h"

#ifdef CUDA_AVAILABLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 获取 CUDA 类型名称 */
const char* cuda_get_type_name(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:  return "float";
        case ACE_DTYPE_FLOAT64:  return "double";
        case ACE_DTYPE_INT32:    return "int";
        case ACE_DTYPE_INT64:    return "long long";
        case ACE_DTYPE_FLOAT16:  return "half";
        case ACE_DTYPE_BFLOAT16: return "__nv_bfloat16";
        case ACE_DTYPE_INT8:     return "signed char";
        case ACE_DTYPE_UINT8:    return "unsigned char";
        case ACE_DTYPE_INT16:    return "short";
        default:                 return "float";
    }
}

/* 获取 CUDA 类型相关的头文件 */
const char* cuda_get_type_headers(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT16:
            return "#include <cuda_fp16.h>\n";
        case ACE_DTYPE_BFLOAT16:
            return "#include <cuda_bf16.h>\n";
        default:
            return "";
    }
}

/* 获取 CUDA 类型转换宏 */
const char* cuda_get_type_macros(ace_dtype_t dtype) {
    /* 不使用宏，由 get_type_helpers 中的内联函数提供 */
    (void)dtype;
    return "";
}

/* 获取 kadd/kmul 宏定义 */
const char* cuda_get_kernel_macros(ace_dtype_t dtype) {
    static char macros_buf[256];
    /* CUDA 原生支持所有类型的运算符，直接展开 */
    snprintf(macros_buf, sizeof(macros_buf),
        "#define kadd(a, b) ((a) + (b))\n"
        "#define kmul(a, b) ((a) * (b))\n");
    return macros_buf;
}

/* 类型辅助函数代码 */
static const char* get_type_helpers(ace_dtype_t dtype) {
    static char helpers_buf[2048];

    if (dtype == ACE_DTYPE_FLOAT16) {
        /* FP16: 使用 float 转换进行运算，避免 __hadd 等弃用函数 */
        snprintf(helpers_buf, sizeof(helpers_buf),
            "/* FP16 类型辅助函数 */\n"
            "typedef __half_raw half_raw;\n"
            "__device__ inline half f32_to_f16(float f) { return __float2half(f); }\n"
            "__device__ inline float f16_to_f32(half h) { return __half2float(h); }\n"
            "__device__ inline half f16_add(half a, half b) { "
            "  return __float2half(__half2float(a) + __half2float(b)); "
            "}\n"
            "__device__ inline half f16_mul(half a, half b) { "
            "  return __float2half(__half2float(a) * __half2float(b)); "
            "}\n"
            "__device__ inline half f16_sub(half a, half b) { "
            "  return __float2half(__half2float(a) - __half2float(b)); "
            "}\n"
            "__device__ inline half f16_div(half a, half b) { "
            "  return __float2half(__half2float(a) / __half2float(b)); "
            "}\n"
            "__device__ inline bool f16_lt(half a, half b) { "
            "  return __half2float(a) < __half2float(b); "
            "}\n"
            "__device__ inline bool f16_le(half a, half b) { "
            "  return __half2float(a) <= __half2float(b); "
            "}\n"
            "__device__ inline bool f16_gt(half a, half b) { "
            "  return __half2float(a) > __half2float(b); "
            "}\n"
            "__device__ inline bool f16_ge(half a, half b) { "
            "  return __half2float(a) >= __half2float(b); "
            "}\n");
        return helpers_buf;
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        /* BF16: 使用 float 转换进行运算 */
        snprintf(helpers_buf, sizeof(helpers_buf),
            "/* BF16 类型辅助函数 */\n"
            "typedef __nv_bfloat16_raw bfloat16_raw;\n"
            "__device__ inline __nv_bfloat16 f32_to_bf16(float f) { return __float2bfloat16(f); }\n"
            "__device__ inline float bf16_to_f32(__nv_bfloat16 h) { return __bfloat162float(h); }\n"
            "__device__ inline __nv_bfloat16 bf16_add(__nv_bfloat16 a, __nv_bfloat16 b) { "
            "  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b)); "
            "}\n"
            "__device__ inline __nv_bfloat16 bf16_mul(__nv_bfloat16 a, __nv_bfloat16 b) { "
            "  return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b)); "
            "}\n"
            "__device__ inline __nv_bfloat16 bf16_sub(__nv_bfloat16 a, __nv_bfloat16 b) { "
            "  return __float2bfloat16(__bfloat162float(a) - __bfloat162float(b)); "
            "}\n"
            "__device__ inline __nv_bfloat16 bf16_div(__nv_bfloat16 a, __nv_bfloat16 b) { "
            "  return __float2bfloat16(__bfloat162float(a) / __bfloat162float(b)); "
            "}\n");
        return helpers_buf;
    }
    return "";
}

char* cuda_translate_code(const char* name, const char* src, ace_dtype_t dtype) {
    const char* type_name = cuda_get_type_name(dtype);
    const char* type_headers = cuda_get_type_headers(dtype);
    const char* type_macros = cuda_get_type_macros(dtype);
    const char* type_helpers = get_type_helpers(dtype);
    const char* kernel_macros = cuda_get_kernel_macros(dtype);

    /* 替换 T 为实际类型 */
    char* code = strdup(src);
    if (!code) return NULL;

    char* p;
    while ((p = strstr(code, "T")) != NULL) {
        int is_type = 1;
        if (p > code) {
            char prev = p[-1];
            if ((prev >= 'a' && prev <= 'z') || (prev >= 'A' && prev <= 'Z') ||
                (prev >= '0' && prev <= '9') || prev == '_') is_type = 0;
        }
        if (p[1]) {
            char next = p[1];
            if ((next >= 'a' && next <= 'z') || (next >= 'A' && next <= 'Z') ||
                (next >= '0' && next <= '9') || next == '_') is_type = 0;
        }

        if (is_type) {
            size_t type_len = strlen(type_name);
            char* new_code = malloc(strlen(code) + type_len + 1);
            if (!new_code) { free(code); return NULL; }

            *p = '\0';
            strcpy(new_code, code);
            strcat(new_code, type_name);
            strcat(new_code, p + 1);
            free(code);
            code = new_code;
        } else {
            p++;
        }
    }

    const char* params_start = strchr(code, '(');
    const char* params_end = strchr(code, ')');
    const char* body_start = strchr(code, '{');
    const char* body_end = strrchr(code, '}');

    if (!params_start || !params_end || !body_start || !body_end) {
        free(code);
        char* out = (char*)malloc(256);
        snprintf(out, 256, "extern \"C\" __global__ void %s() {}\n", name);
        return out;
    }

    size_t params_len = params_end - params_start + 1;
    char* params = (char*)malloc(params_len + 1);
    strncpy(params, params_start, params_len);
    params[params_len] = '\0';

    size_t body_len = body_end - body_start - 1;

    size_t total_len = strlen(name) + params_len + body_len + 1024 +
                       strlen(type_headers) + strlen(type_macros) + strlen(type_helpers) + strlen(kernel_macros);
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
        "%s"
        "%s"
        "%s"
        "%s"
        "extern \"C\" __global__ void %s%s\n"
        "{\n"
        "    const int GID = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    const int LID = threadIdx.x;\n"
        "    const int BSIZE = blockDim.x;\n"
        "    %.*s\n"
        "}\n",
        type_headers,
        kernel_macros,
        type_helpers,
        type_macros,
        name, params,
        (int)body_len, body_start + 1
    );

    free(params);
    free(code);
    return out;
}

#endif /* CUDA_AVAILABLE */
