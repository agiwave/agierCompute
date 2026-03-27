/**
 * @file opencl_type_utils.c
 * @brief OpenCL type translation utilities
 */
#include "opencl_backend.h"

#ifdef OPENCL_AVAILABLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 获取 OpenCL 类型名称 */
const char* ocl_get_type_name(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:  return "float";
        case ACE_DTYPE_FLOAT64:  return "double";
        case ACE_DTYPE_INT32:    return "int";
        case ACE_DTYPE_INT64:    return "long";
        case ACE_DTYPE_FLOAT16:  return "half";
        case ACE_DTYPE_BFLOAT16: return "ushort";
        case ACE_DTYPE_INT8:     return "char";
        case ACE_DTYPE_UINT8:    return "uchar";
        case ACE_DTYPE_INT16:    return "short";
        default:                 return "float";
    }
}

/* 获取 OpenCL 扩展 pragma
 * 根据设备能力返回扩展声明或空字符串（使用模拟）
 */
const char* ocl_get_extension(ace_dtype_t dtype) {
    /* 声明外部变量 */
    extern ocl_device_extensions_t g_device_exts;
    
    switch (dtype) {
        case ACE_DTYPE_FLOAT16:
            /* 检查设备是否支持 cl_khr_fp16 */
            if (g_device_exts.has_fp16) {
                return "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
            } else {
                return "";  /* 使用 uint 模拟 */
            }
        case ACE_DTYPE_FLOAT64:
            if (g_device_exts.has_fp64) {
                return "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
            } else {
                return "";  /* 使用 float 模拟 */
            }
        default:
            return "";
    }
}

/* 获取 kadd/kmul 等内核函数宏和常量定义 */
const char* ocl_get_kernel_macros(ace_dtype_t dtype) {
    static char macros_buf[1024];
    extern ocl_device_extensions_t g_device_exts;
    
    if (dtype == ACE_DTYPE_FLOAT16 && !g_device_exts.has_fp16) {
        /* FP16 模拟模式 */
        snprintf(macros_buf, sizeof(macros_buf),
            "/* FP16 模拟内核函数宏 */\n"
            "float f16_to_f32(uint x) {\n"
            "    uint sign = (x >> 15u) & 0x1u;\n"
            "    uint exp = (x >> 10u) & 0x1Fu;\n"
            "    uint man = x & 0x3FFu;\n"
            "    if (exp == 0u) return sign != 0u ? -0.0f : 0.0f;\n"
            "    if (exp == 31u) return sign != 0u ? -1.0f/0.0f : 1.0f/0.0f;\n"
            "    uint result = (sign << 31u) | ((exp + 112u) << 23u) | (man << 13u);\n"
            "    return as_float(result);\n"
            "}\n"
            "uint f32_to_f16(float x) {\n"
            "    uint u = as_uint(x);\n"
            "    uint sign = (u >> 16u) & 0x8000u;\n"
            "    uint exp = ((u >> 23u) & 0xFFu) - 112u;\n"
            "    uint man = (u >> 13u) & 0x3FFu;\n"
            "    if ((u & 0x7FFFFFFFu) == 0u) return sign;\n"
            "    if (exp > 30u) return sign | 0x7C00u;\n"
            "    return sign | (exp << 10u) | man;\n"
            "}\n"
            "uint f16_add(uint a, uint b) { return f32_to_f16(f16_to_f32(a) + f16_to_f32(b)); }\n"
            "uint f16_mul(uint a, uint b) { return f32_to_f16(f16_to_f32(a) * f16_to_f32(b)); }\n"
            "#define kadd(a, b) f16_add(a, b)\n"
            "#define kmul(a, b) f16_mul(a, b)\n"
            "#define K_ZERO 0u\n"
            "#define K_ONE 0x3C00u\n");
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        /* BF16 模拟模式 */
        snprintf(macros_buf, sizeof(macros_buf),
            "/* BF16 模拟内核函数宏 */\n"
            "float bf16_to_f32(uint x) {\n"
            "    uint sign = (x >> 15u) & 0x1u;\n"
            "    uint exp = (x >> 7u) & 0xFFu;\n"
            "    uint man = x & 0x7Fu;\n"
            "    if (exp == 0u) return sign != 0u ? -0.0f : 0.0f;\n"
            "    if (exp == 255u) return sign != 0u ? -1.0f/0.0f : 1.0f/0.0f;\n"
            "    uint result = (sign << 31u) | ((exp + 112u) << 23u) | (man << 16u);\n"
            "    return as_float(result);\n"
            "}\n"
            "uint f32_to_bf16(float x) {\n"
            "    uint u = as_uint(x);\n"
            "    uint sign = (u >> 16u) & 0x8000u;\n"
            "    uint exp = ((u >> 23u) & 0xFFu) - 112u;\n"
            "    uint man = (u >> 16u) & 0x7Fu;\n"
            "    if ((u & 0x7FFFFFFFu) == 0u) return sign;\n"
            "    if (exp > 254u) return sign | 0x7F80u;\n"
            "    return sign | (exp << 7u) | man;\n"
            "}\n"
            "uint bf16_add(uint a, uint b) { return f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b)); }\n"
            "uint bf16_mul(uint a, uint b) { return f32_to_bf16(bf16_to_f32(a) * bf16_to_f32(b)); }\n"
            "#define kadd(a, b) bf16_add(a, b)\n"
            "#define kmul(a, b) bf16_mul(a, b)\n"
            "#define K_ZERO 0u\n"
            "#define K_ONE 0x3F80u\n");
    } else if (dtype == ACE_DTYPE_INT8 || dtype == ACE_DTYPE_UINT8) {
        snprintf(macros_buf, sizeof(macros_buf),
            "/* INT8/UINT8 内核函数宏 */\n"
            "#define kadd(a, b) (((a) + (b)) & 0xFFu)\n"
            "#define kmul(a, b) (((a) * (b)) & 0xFFu)\n"
            "#define K_ZERO 0u\n"
            "#define K_ONE 1u\n");
    } else if (dtype == ACE_DTYPE_INT16) {
        snprintf(macros_buf, sizeof(macros_buf),
            "/* INT16 内核函数宏 */\n"
            "#define kadd(a, b) (((a) + (b)) & 0xFFFFu)\n"
            "#define kmul(a, b) (((a) * (b)) & 0xFFFFu)\n"
            "#define K_ZERO 0u\n"
            "#define K_ONE 1u\n");
    } else {
        /* 原生类型 */
        snprintf(macros_buf, sizeof(macros_buf),
            "/* 原生类型内核函数宏 */\n"
            "#define kadd(a, b) ((a) + (b))\n"
            "#define ksub(a, b) ((a) - (b))\n"
            "#define kmul(a, b) ((a) * (b))\n"
            "#define kdiv(a, b) ((a) / (b))\n"
            "#define klt(a, b) ((a) < (b))\n"
            "#define kle(a, b) ((a) <= (b))\n"
            "#define kgt(a, b) ((a) > (b))\n"
            "#define kge(a, b) ((a) >= (b))\n"
            "#define keq(a, b) ((a) == (b))\n"
            "#define kne(a, b) ((a) != (b))\n"
            "#define K_ZERO (%s)0\n"
            "#define K_ONE (%s)1\n"
            "#define K_NEG_ONE (%s)-1\n",
            ocl_get_type_name(dtype),
            ocl_get_type_name(dtype),
            ocl_get_type_name(dtype));
    }
    return macros_buf;
}

/* BF16 辅助函数代码 */
static const char* get_bf16_helpers(void) {
    static char bf16_buf[1024];
    snprintf(bf16_buf, sizeof(bf16_buf),
        "/* BF16 转换函数 */\n"
        "float bf16_to_f32(ushort x) {\n"
        "    uint sign = (x >> 15) & 0x1;\n"
        "    uint exp = (x >> 7) & 0xFF;\n"
        "    uint man = x & 0x7F;\n"
        "    if (exp == 0) return sign != 0 ? -0.0f : 0.0f;\n"
        "    if (exp == 255) return sign != 0 ? -INFINITY : INFINITY;\n"
        "    uint result = (sign << 31) | ((exp + 112) << 23) | (man << 16);\n"
        "    return as_float(result);\n"
        "}\n"
        "ushort f32_to_bf16(float x) {\n"
        "    uint u = as_uint(x);\n"
        "    ushort sign = (ushort)((u >> 16) & 0x8000);\n"
        "    ushort exp = (ushort)(((u >> 23) & 0xFF) - 112);\n"
        "    ushort man = (ushort)((u >> 16) & 0x7F);\n"
        "    if ((u & 0x7FFFFFFF) == 0) return sign;\n"
        "    if (exp < 0) return sign;\n"
        "    if (exp > 254) return sign | 0x7F80;\n"
        "    return sign | (exp << 7) | man;\n"
        "}\n"
        "ushort bf16_add(ushort a, ushort b) { return f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b)); }\n"
        "ushort bf16_mul(ushort a, ushort b) { return f32_to_bf16(bf16_to_f32(a) * bf16_to_f32(b)); }\n");
    return bf16_buf;
}

char* ocl_translate_code(const char* name, const char* src, ace_dtype_t dtype) {
    const char* type_name = ocl_get_type_name(dtype);
    const char* extension = ocl_get_extension(dtype);
    const char* kernel_macros = ocl_get_kernel_macros(dtype);

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

    /* 添加 __global 到指针参数 */
    char* with_global = NULL;
    char* ptr = strstr(code, "(");
    if (ptr) {
        char* end = strchr(ptr, ')');
        if (end) {
            size_t prefix_len = ptr - code + 1;
            with_global = malloc(strlen(code) + 100);
            if (!with_global) { free(code); return NULL; }

            memcpy(with_global, code, prefix_len);
            char* dst = with_global + prefix_len;
            char* param_start = ptr + 1;

            while (param_start < end) {
                while (param_start < end && (*param_start == ' ' || *param_start == '\t')) param_start++;
                if (param_start >= end) break;

                char* comma = NULL;
                char* search = param_start;
                int paren_depth = 0;
                while (search < end) {
                    if (*search == '(') paren_depth++;
                    else if (*search == ')') paren_depth--;
                    else if (*search == ',' && paren_depth == 0) {
                        comma = search;
                        break;
                    }
                    search++;
                }
                if (!comma) comma = end;

                char* star = NULL;
                for (char* s = param_start; s < comma; s++) {
                    if (*s == '*') {
                        star = s;
                        break;
                    }
                }

                if (star) {
                    memcpy(dst, "__global ", 9);
                    dst += 9;
                    memcpy(dst, param_start, comma - param_start);
                    dst += comma - param_start;
                } else {
                    memcpy(dst, param_start, comma - param_start);
                    dst += comma - param_start;
                }

                param_start = comma;
                if (param_start < end && *param_start == ',') {
                    *dst++ = ',';
                    param_start++;
                }
            }

            strcpy(dst, end);
            free(code);
            code = with_global;
        }
    }

    const char* params_start = strchr(code, '(');
    const char* params_end = strchr(code, ')');
    const char* body_start = strchr(code, '{');
    const char* body_end = strrchr(code, '}');

    if (!params_start || !params_end || !body_start || !body_end) {
        free(code);
        char* out = (char*)malloc(256);
        snprintf(out, 256, "%s__kernel void %s() { int GID = get_global_id(0); }\n",
                 extension, name);
        return out;
    }

    size_t params_len = params_end - params_start + 1;
    char* params = (char*)malloc(params_len + 1);
    strncpy(params, params_start, params_len);
    params[params_len] = '\0';

    size_t body_len = body_end - body_start - 1;

    const char* bf16_helpers = (dtype == ACE_DTYPE_BFLOAT16) ? get_bf16_helpers() : "";

    size_t total_len = strlen(name) + params_len + body_len + 1024 +
                       strlen(extension) + strlen(bf16_helpers) + strlen(kernel_macros);
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
        "%s"
        "%s"
        "%s"
        "__kernel void %s%s\n"
        "{\n"
        "    int GID = get_global_id(0);\n"
        "    int LID = get_local_id(0);\n"
        "    int BSIZE = get_local_size(0);\n"
        "    %.*s\n"
        "}\n",
        extension,
        kernel_macros,
        bf16_helpers,
        name, params,
        (int)body_len, body_start + 1
    );

    free(params);
    free(code);
    return out;
}

#endif /* OPENCL_AVAILABLE */
