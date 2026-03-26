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

/* 获取 OpenCL 扩展 pragma */
const char* ocl_get_extension(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT16:
            return "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        case ACE_DTYPE_FLOAT64:
            return "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        default:
            return "";
    }
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
                       strlen(extension) + strlen(bf16_helpers);
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
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
        bf16_helpers,
        name, params,
        (int)body_len, body_start + 1
    );

    free(params);
    free(code);
    return out;
}

#endif /* OPENCL_AVAILABLE */
