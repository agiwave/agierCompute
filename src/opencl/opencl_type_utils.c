/**
 * @file opencl_type_utils.c
 * @brief OpenCL 类型辅助工具和动态 kxxx 函数注入
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "ace.h"

/* 设备扩展信息（由 opencl_device.c 提供） */
typedef struct {
    int has_fp16;
    int has_fp64;
    int has_int64;
} ocl_device_extensions_t;
extern ocl_device_extensions_t g_device_exts;

/* ============================================================================
 * 类型名称映射 - 根据设备能力选择原生或模拟
 * ============================================================================ */

const char* ocl_get_type_name(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:  return "float";
        case ACE_DTYPE_FLOAT64:  return "double";
        case ACE_DTYPE_INT32:    return "int";
        case ACE_DTYPE_INT64:    return "long";
        case ACE_DTYPE_FLOAT16:
            return g_device_exts.has_fp16 ? "half" : "ushort";
        case ACE_DTYPE_BFLOAT16:
            return "ushort";  /* BF16 总是使用 ushort 模拟 */
        case ACE_DTYPE_INT8:     return "char";
        case ACE_DTYPE_UINT8:    return "uchar";
        case ACE_DTYPE_INT16:    return "short";
        default:                 return "float";
    }
}

/* ============================================================================
 * 扩展声明
 * ============================================================================ */

const char* ocl_get_extension(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT16:
            return g_device_exts.has_fp16 ? "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" : "";
        case ACE_DTYPE_FLOAT64:
            return g_device_exts.has_fp64 ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" : "";
        default:
            return "";
    }
}

/* ============================================================================
 * 只包含类型转换函数，运算函数动态生成
 * ============================================================================ */

static const char* get_type_converters(ace_dtype_t dtype) {
    static char conv_buf[4096];
    
    if (dtype == ACE_DTYPE_FLOAT16 && !g_device_exts.has_fp16) {
        /* FP16 模拟模式 - 只定义转换函数 */
        snprintf(conv_buf, sizeof(conv_buf),
            "/* FP16 类型转换函数 */\n"
            "float f16_to_f32(ushort x) {\n"
            "    uint sign = (x >> 15u) & 0x1u;\n"
            "    uint exp = (x >> 10u) & 0x1Fu;\n"
            "    uint man = x & 0x3FFu;\n"
            "    if (exp == 0u) return sign != 0u ? -0.0f : 0.0f;\n"
            "    if (exp == 31u) return sign != 0u ? -1.0f/0.0f : 1.0f/0.0f;\n"
            "    uint result = (sign << 31u) | ((exp + 112u) << 23u) | (man << 13u);\n"
            "    return as_float(result);\n"
            "}\n"
            "ushort f32_to_f16(float x) {\n"
            "    uint u = as_uint(x);\n"
            "    uint sign = (u >> 16u) & 0x8000u;\n"
            "    uint exp = ((u >> 23u) & 0xFFu) - 112u;\n"
            "    uint man = (u >> 13u) & 0x3FFu;\n"
            "    if ((u & 0x7FFFFFFFu) == 0u) return sign;\n"
            "    if (exp > 30u) return sign | 0x7C00u;\n"
            "    return sign | (exp << 10u) | man;\n"
            "}\n");
        return conv_buf;
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        /* BF16 模拟模式 - 只定义转换函数 */
        snprintf(conv_buf, sizeof(conv_buf),
            "/* BF16 类型转换函数 */\n"
            "float bf16_to_f32(ushort x) {\n"
            "    uint sign = (x >> 15u) & 0x1u;\n"
            "    uint exp = (x >> 7u) & 0xFFu;\n"
            "    uint man = x & 0x7Fu;\n"
            "    if (exp == 0u) return sign != 0u ? -0.0f : 0.0f;\n"
            "    if (exp == 255u) return sign != 0u ? -1.0f/0.0f : 1.0f/0.0f;\n"
            "    uint result = (sign << 31u) | (exp << 23u) | (man << 16u);\n"
            "    return as_float(result);\n"
            "}\n"
            "ushort f32_to_bf16(float x) {\n"
            "    uint u = as_uint(x);\n"
            "    uint sign = (u >> 16u) & 0x8000u;\n"
            "    uint exp = (u >> 23u) & 0xFFu;\n"
            "    uint man = (u >> 16u) & 0x7Fu;\n"
            "    if ((u & 0x7FFFFFFFu) == 0u) return sign;\n"
            "    if (exp == 255u) return sign | 0x7F80u;\n"
            "    return sign | (exp << 7u) | man;\n"
            "}\n");
        return conv_buf;
    }
    return "";
}

/* ============================================================================
 * 动态 kxxx 函数注入
 * ============================================================================ */

static void inject_used_k_functions(char* out, size_t out_size, const char* code, ace_dtype_t dtype) {
    int is_fp16_sim = (dtype == ACE_DTYPE_FLOAT16 && !g_device_exts.has_fp16);
    int is_bf16 = (dtype == ACE_DTYPE_BFLOAT16);
    
    if (!is_fp16_sim && !is_bf16) {
        return;  /* 只有模拟模式需要注入 */
    }

    char inject_buf[8192] = "";
    char* p = inject_buf;
    char* end = inject_buf + sizeof(inject_buf) - 1;

    const char* type_name = "ushort";
    const char* to_f32 = is_bf16 ? "bf16_to_f32" : "f16_to_f32";
    const char* from_f32 = is_bf16 ? "f32_to_bf16" : "f32_to_f16";

    /* 检测并注入 kadd */
    if (strstr(code, "kadd")) {
        int len = snprintf(p, end - p,
            "ushort kadd(ushort a, ushort b) { "
            "  return %s(%s(a) + %s(b)); "
            "}\n",
            from_f32, to_f32, to_f32);
        p += len;
    }

    /* 检测并注入 ksub */
    if (strstr(code, "ksub")) {
        int len = snprintf(p, end - p,
            "ushort ksub(ushort a, ushort b) { "
            "  return %s(%s(a) - %s(b)); "
            "}\n",
            from_f32, to_f32, to_f32);
        p += len;
    }

    /* 检测并注入 kmul */
    if (strstr(code, "kmul")) {
        int len = snprintf(p, end - p,
            "ushort kmul(ushort a, ushort b) { "
            "  return %s(%s(a) * %s(b)); "
            "}\n",
            from_f32, to_f32, to_f32);
        p += len;
    }

    /* 检测并注入 kdiv */
    if (strstr(code, "kdiv")) {
        int len = snprintf(p, end - p,
            "ushort kdiv(ushort a, ushort b) { "
            "  return %s(%s(a) / %s(b)); "
            "}\n",
            from_f32, to_f32, to_f32);
        p += len;
    }

    /* 检测并注入比较函数 */
    if (strstr(code, "klt")) {
        int len = snprintf(p, end - p,
            "bool klt(ushort a, ushort b) { "
            "  return %s(a) < %s(b); "
            "}\n",
            to_f32, to_f32);
        p += len;
    }
    if (strstr(code, "kle")) {
        int len = snprintf(p, end - p,
            "bool kle(ushort a, ushort b) { "
            "  return %s(a) <= %s(b); "
            "}\n",
            to_f32, to_f32);
        p += len;
    }
    if (strstr(code, "kgt")) {
        int len = snprintf(p, end - p,
            "bool kgt(ushort a, ushort b) { "
            "  return %s(a) > %s(b); "
            "}\n",
            to_f32, to_f32);
        p += len;
    }
    if (strstr(code, "kge")) {
        int len = snprintf(p, end - p,
            "bool kge(ushort a, ushort b) { "
            "  return %s(a) >= %s(b); "
            "}\n",
            to_f32, to_f32);
        p += len;
    }

    /* 将注入的函数插入到输出代码中 */
    if (inject_buf[0] != '\0') {
        char* temp = (char*)malloc(strlen(out) + strlen(inject_buf) + 10);
        if (temp) {
            strcpy(temp, inject_buf);
            strcat(temp, "\n");
            strcat(temp, out);
            strcpy(out, temp);
            free(temp);
        }
    }
}

/* ============================================================================
 * 内核函数宏（只定义宏，具体函数动态注入）
 * ============================================================================ */

const char* ocl_get_kernel_macros(ace_dtype_t dtype) {
    static char macros_buf[4096];
    
    if (dtype == ACE_DTYPE_FLOAT16 && !g_device_exts.has_fp16) {
        snprintf(macros_buf, sizeof(macros_buf),
            "/* FP16 模拟 - kxxx 函数动态注入 */\n"
            "#define K_ZERO (ushort)0u\n"
            "#define K_ONE (ushort)0x3C00u\n"
            "#define K_NEG_ONE (ushort)0xBC00u\n");
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        snprintf(macros_buf, sizeof(macros_buf),
            "/* BF16 模拟 - kxxx 函数动态注入 */\n"
            "#define K_ZERO (ushort)0u\n"
            "#define K_ONE (ushort)0x3F80u\n"
            "#define K_NEG_ONE (ushort)0xBF80u\n");
    } else if (dtype == ACE_DTYPE_INT8 || dtype == ACE_DTYPE_UINT8) {
        snprintf(macros_buf, sizeof(macros_buf),
            "/* INT8/UINT8 内核函数宏 */\n"
            "#define kadd(a, b) (((a) + (b)) & 0xFFu)\n"
            "#define ksub(a, b) (((a) - (b)) & 0xFFu)\n"
            "#define kmul(a, b) (((a) * (b)) & 0xFFu)\n"
            "#define kdiv(a, b) (((a) / (b)) & 0xFFu)\n"
            "#define klt(a, b) ((a) < (b))\n"
            "#define kle(a, b) ((a) <= (b))\n"
            "#define kgt(a, b) ((a) > (b))\n"
            "#define kge(a, b) ((a) >= (b))\n"
            "#define keq(a, b) ((a) == (b))\n"
            "#define kne(a, b) ((a) != (b))\n"
            "#define K_ZERO (uchar)0u\n"
            "#define K_ONE (uchar)1u\n"
            "#define K_NEG_ONE (uchar)255u\n");
    } else if (dtype == ACE_DTYPE_INT16) {
        snprintf(macros_buf, sizeof(macros_buf),
            "/* INT16 内核函数宏 */\n"
            "#define kadd(a, b) (((a) + (b)) & 0xFFFFu)\n"
            "#define ksub(a, b) (((a) - (b)) & 0xFFFFu)\n"
            "#define kmul(a, b) (((a) * (b)) & 0xFFFFu)\n"
            "#define kdiv(a, b) (((a) / (b)) & 0xFFFFu)\n"
            "#define klt(a, b) ((a) < (b))\n"
            "#define kle(a, b) ((a) <= (b))\n"
            "#define kgt(a, b) ((a) > (b))\n"
            "#define kge(a, b) ((a) >= (b))\n"
            "#define keq(a, b) ((a) == (b))\n"
            "#define kne(a, b) ((a) != (b))\n"
            "#define K_ZERO (ushort)0u\n"
            "#define K_ONE (ushort)1u\n"
            "#define K_NEG_ONE (ushort)65535u\n");
    } else if (dtype == ACE_DTYPE_FLOAT16 && g_device_exts.has_fp16) {
        snprintf(macros_buf, sizeof(macros_buf),
            "/* FP16 原生模式 */\n"
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
            "#define K_ZERO half(0)\n"
            "#define K_ONE half(1)\n"
            "#define K_NEG_ONE half(-1)\n");
    } else {
        const char* type_name = ocl_get_type_name(dtype);
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
            type_name, type_name, type_name);
    }
    return macros_buf;
}

/* ============================================================================
 * 代码翻译主函数
 * ============================================================================ */

char* ocl_translate_code(const char* name, const char* src, ace_dtype_t dtype) {
    const char* type_name = ocl_get_type_name(dtype);
    const char* extension = ocl_get_extension(dtype);
    const char* converters = get_type_converters(dtype);
    const char* kernel_macros = ocl_get_kernel_macros(dtype);

    /* 替换 T 为实际类型 */
    char* code = strdup(src);
    if (!code) return NULL;

    char* p = code;
    while ((p = strstr(p, "T")) != NULL) {
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
            p = new_code + strlen(code) - strlen(p + 1);
        } else {
            p++;
        }
    }

    /* 添加 __global 指针修饰符 */
    p = strchr(code, '(');
    if (p) {
        char* end = strchr(p, ')');
        if (end) {
            size_t len = strlen(code) + 64;
            char* with_global = malloc(len);
            char* dst = with_global;
            memcpy(dst, code, p - code + 1);
            dst += p - code + 1;

            char* param_start = p + 1;
            while (param_start < end) {
                while (param_start < end && (*param_start == ' ' || *param_start == '\t')) param_start++;
                
                while (param_start < end && *param_start != ',' && *param_start != ')') {
                    if (*param_start == '*') {
                        memcpy(dst, " __global ", 10);  /* 前后都有空格 */
                        dst += 10;
                    }
                    *dst++ = *param_start++;
                }
                
                if (param_start < end && *param_start == ',') {
                    *dst++ = *param_start++;
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

    /* 动态检测内核代码中使用的 kxxx 函数 */
    int need_converters = 0;
    int is_sim = (dtype == ACE_DTYPE_FLOAT16 && !g_device_exts.has_fp16) || 
                 (dtype == ACE_DTYPE_BFLOAT16);
    if (is_sim) {
        if (strstr(code, "kadd") || strstr(code, "ksub") || strstr(code, "kmul") ||
            strstr(code, "kdiv") || strstr(code, "klt") || strstr(code, "kle") ||
            strstr(code, "kgt") || strstr(code, "kge") || strstr(code, "keq") ||
            strstr(code, "kne")) {
            need_converters = 1;
        }
    }

    size_t conv_len = need_converters ? strlen(converters) : 0;
    size_t macros_len = strlen(kernel_macros);
    size_t total_len = strlen(name) + params_len + body_len + 2048 +
                       strlen(extension) + conv_len + macros_len;
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
        "%s"
        "%s"
        "%s"
        "\n__kernel void %s%s\n"
        "{\n"
        "    int GID = get_global_id(0);\n"
        "    int LID = get_local_id(0);\n"
        "    int BSIZE = get_local_size(0);\n"
        "    %.*s\n"
        "}\n",
        extension,
        need_converters ? converters : "",
        kernel_macros,
        name, params,
        (int)body_len, body_start + 1
    );

    /* 动态注入实际使用到的 kxxx 函数 */
    if (need_converters) {
        inject_used_k_functions(out, total_len, code, dtype);
    }

    free(params);
    free(code);
    return out;
}
