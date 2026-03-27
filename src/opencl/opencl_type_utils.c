/**
 * @file opencl_type_utils.c
 * @brief OpenCL 类型辅助工具和动态 kxxx 函数注入
 * 
 * 编译流程:
 * 1. 扫描内核代码，找到所有 kxxx( 模式的函数调用
 * 2. 遍历所有找到的 kxxx 函数
 * 3. 对每个 kxxx，检测类型是否原生支持
 * 4. 原生支持：注入 #define kxxx(a,b) ((a) op (b))
 * 5. 非原生支持：
 *    - 如果类型定义未注入：先注入类型定义和转换函数
 *    - 注入 kxxx 的模拟实现函数
 * 6. 编译最终的内核字符串
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
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
 * 设备能力查询
 * ============================================================================ */

static int ocl_supports_native_kfunc(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:
        case ACE_DTYPE_INT32:
        case ACE_DTYPE_INT8:
        case ACE_DTYPE_UINT8:
        case ACE_DTYPE_INT16:
            return 1;  /* 原生支持 */
        case ACE_DTYPE_FLOAT64:
            return g_device_exts.has_fp64;
        case ACE_DTYPE_FLOAT16:
            return g_device_exts.has_fp16;
        case ACE_DTYPE_BFLOAT16:
            return 0;  /* BF16 总是需要模拟 */
        case ACE_DTYPE_INT64:
            return g_device_exts.has_int64;
        default:
            return 1;
    }
}

/* ============================================================================
 * 类型名称和扩展
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
            return "ushort";
        case ACE_DTYPE_INT8:     return "char";
        case ACE_DTYPE_UINT8:    return "uchar";
        case ACE_DTYPE_INT16:    return "short";
        default:                 return "float";
    }
}

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

const char* ocl_get_type_macros(ace_dtype_t dtype) {
    static char macros_buf[1024];
    const char* type_name = ocl_get_type_name(dtype);
    snprintf(macros_buf, sizeof(macros_buf),
        "#define K_ZERO (%s)0\n"
        "#define K_ONE (%s)1\n"
        "#define K_NEG_ONE (%s)-1\n",
        type_name, type_name, type_name);
    return macros_buf;
}

/* ============================================================================
 * 类型定义和转换函数（用于非原生支持的类型）
 * ============================================================================ */

static const char* get_type_definition(ace_dtype_t dtype) {
    static char def_buf[4096];

    if (dtype == ACE_DTYPE_FLOAT16 && !g_device_exts.has_fp16) {
        snprintf(def_buf, sizeof(def_buf),
            "/* FP16 模拟类型定义和转换函数 */\n"
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
        return def_buf;
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        snprintf(def_buf, sizeof(def_buf),
            "/* BF16 模拟类型定义和转换函数 */\n"
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
        return def_buf;
    }
    return "";
}

/* ============================================================================
 * 核心注入函数：注入单个 kxxx 函数
 * ============================================================================ */

static int inject_k_function_impl(char* buf, const char* func_name, ace_dtype_t dtype, int is_native) {
    if (is_native) {
        /* 原生支持：注入宏定义 */
        if (strcmp(func_name, "kadd") == 0)
            return sprintf(buf, "#define kadd(a, b) ((a) + (b))\n");
        if (strcmp(func_name, "ksub") == 0)
            return sprintf(buf, "#define ksub(a, b) ((a) - (b))\n");
        if (strcmp(func_name, "kmul") == 0)
            return sprintf(buf, "#define kmul(a, b) ((a) * (b))\n");
        if (strcmp(func_name, "kdiv") == 0)
            return sprintf(buf, "#define kdiv(a, b) ((a) / (b))\n");
        if (strcmp(func_name, "klt") == 0)
            return sprintf(buf, "#define klt(a, b) ((a) < (b))\n");
        if (strcmp(func_name, "kle") == 0)
            return sprintf(buf, "#define kle(a, b) ((a) <= (b))\n");
        if (strcmp(func_name, "kgt") == 0)
            return sprintf(buf, "#define kgt(a, b) ((a) > (b))\n");
        if (strcmp(func_name, "kge") == 0)
            return sprintf(buf, "#define kge(a, b) ((a) >= (b))\n");
        if (strcmp(func_name, "keq") == 0)
            return sprintf(buf, "#define keq(a, b) ((a) == (b))\n");
        if (strcmp(func_name, "kne") == 0)
            return sprintf(buf, "#define kne(a, b) ((a) != (b))\n");
    } else {
        /* 非原生支持：注入模拟函数 */
        const char* type_name = ocl_get_type_name(dtype);
        const char* to_f32 = (dtype == ACE_DTYPE_FLOAT16) ? "f16_to_f32" : "bf16_to_f32";
        const char* from_f32 = (dtype == ACE_DTYPE_FLOAT16) ? "f32_to_f16" : "f32_to_bf16";
        
        if (strcmp(func_name, "kadd") == 0)
            return sprintf(buf, "%s kadd(%s a, %s b) { return %s(%s(a) + %s(b)); }\n",
                          type_name, type_name, type_name, from_f32, to_f32, to_f32);
        if (strcmp(func_name, "ksub") == 0)
            return sprintf(buf, "%s ksub(%s a, %s b) { return %s(%s(a) - %s(b)); }\n",
                          type_name, type_name, type_name, from_f32, to_f32, to_f32);
        if (strcmp(func_name, "kmul") == 0)
            return sprintf(buf, "%s kmul(%s a, %s b) { return %s(%s(a) * %s(b)); }\n",
                          type_name, type_name, type_name, from_f32, to_f32, to_f32);
        if (strcmp(func_name, "kdiv") == 0)
            return sprintf(buf, "%s kdiv(%s a, %s b) { return %s(%s(a) / %s(b)); }\n",
                          type_name, type_name, type_name, from_f32, to_f32, to_f32);
        if (strcmp(func_name, "klt") == 0)
            return sprintf(buf, "bool klt(%s a, %s b) { return %s(a) < %s(b); }\n",
                          type_name, type_name, to_f32, to_f32);
        if (strcmp(func_name, "kle") == 0)
            return sprintf(buf, "bool kle(%s a, %s b) { return %s(a) <= %s(b); }\n",
                          type_name, type_name, to_f32, to_f32);
        if (strcmp(func_name, "kgt") == 0)
            return sprintf(buf, "bool kgt(%s a, %s b) { return %s(a) > %s(b); }\n",
                          type_name, type_name, to_f32, to_f32);
        if (strcmp(func_name, "kge") == 0)
            return sprintf(buf, "bool kge(%s a, %s b) { return %s(a) >= %s(b); }\n",
                          type_name, type_name, to_f32, to_f32);
        if (strcmp(func_name, "keq") == 0)
            return sprintf(buf, "bool keq(%s a, %s b) { return %s(a) == %s(b); }\n",
                          type_name, type_name, to_f32, to_f32);
        if (strcmp(func_name, "kne") == 0)
            return sprintf(buf, "bool kne(%s a, %s b) { return %s(a) != %s(b); }\n",
                          type_name, type_name, to_f32, to_f32);
    }
    return 0;
}

/* ============================================================================
 * 扫描代码并注入所有找到的 kxxx 函数
 * ============================================================================ */

static void scan_and_inject(char* out, const char* code, ace_dtype_t dtype,
                            int* injected_mask, int* type_def_injected) {
    int is_native = ocl_supports_native_kfunc(dtype);
    
    /* 非原生支持：先注入类型定义（如果还未注入） */
    if (!is_native && !*type_def_injected) {
        const char* type_def = get_type_definition(dtype);
        if (type_def && type_def[0]) {
            char* temp = (char*)malloc(strlen(out) + strlen(type_def) + 10);
            if (temp) {
                strcpy(temp, type_def);
                strcat(temp, "\n");
                strcat(temp, out);
                strcpy(out, temp);
                free(temp);
                *type_def_injected = 1;
            }
        }
    }
    
    char inject_buf[8192] = "";
    char* p = inject_buf;
    char* end = inject_buf + sizeof(inject_buf) - 1;

    const char* s = code;
    while (*s) {
        if (islower(*s) && s[1] && islower(s[1]) && s[2] && islower(s[2])) {
            const char* start = s;
            while (islower(*s)) s++;

            if (*s == '(') {
                char func_name[16];
                size_t len = s - start;
                if (len < sizeof(func_name)) {
                    strncpy(func_name, start, len);
                    func_name[len] = '\0';

                    if (func_name[0] == 'k' && func_name[1] && func_name[2]) {
                        int func_id = -1;
                        if (strcmp(func_name, "kadd") == 0) func_id = 0;
                        else if (strcmp(func_name, "ksub") == 0) func_id = 1;
                        else if (strcmp(func_name, "kmul") == 0) func_id = 2;
                        else if (strcmp(func_name, "kdiv") == 0) func_id = 3;
                        else if (strcmp(func_name, "klt") == 0) func_id = 4;
                        else if (strcmp(func_name, "kle") == 0) func_id = 5;
                        else if (strcmp(func_name, "kgt") == 0) func_id = 6;
                        else if (strcmp(func_name, "kge") == 0) func_id = 7;
                        else if (strcmp(func_name, "keq") == 0) func_id = 8;
                        else if (strcmp(func_name, "kne") == 0) func_id = 9;

                        if (func_id >= 0 && func_id < 10 && !(*injected_mask & (1 << func_id))) {
                            int inj_len = inject_k_function_impl(p, func_name, dtype, is_native);
                            if (inj_len > 0) {
                                p += inj_len;
                                *injected_mask |= (1 << func_id);
                            }
                        }
                    }
                }
            }
        } else {
            s++;
        }
    }

    if (p > inject_buf) {
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
 * 代码翻译主函数
 * ============================================================================ */

char* ocl_translate_code(const char* name, const char* src, ace_dtype_t dtype) {
    const char* type_name = ocl_get_type_name(dtype);
    const char* extension = ocl_get_extension(dtype);
    const char* type_macros = ocl_get_type_macros(dtype);

    /* 步骤 1: 替换 T 为实际类型 */
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
                        memcpy(dst, " __global ", 10);
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

    /* 步骤 2: 扫描代码，查找有没有 kxxx 函数调用 */
    int has_k_functions = 0;
    const char* s = code;
    while (*s) {
        if (s[0] == 'k' && islower(s[1]) && islower(s[2])) {
            const char* start = s;
            while (islower(*s)) s++;
            if (*s == '(' && (s - start) <= 5) {
                has_k_functions = 1;
                break;
            }
        } else {
            s++;
        }
    }

    /* 步骤 3: 构建基础代码框架 */
    size_t total_len = strlen(name) + params_len + body_len + 2048 +
                       strlen(extension) + strlen(type_macros);
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
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
        type_macros,
        name, params,
        (int)body_len, body_start + 1
    );

    /* 步骤 4-6: 遍历所有 kxxx，注入对应的实现 */
    if (has_k_functions) {
        int injected_mask = 0;
        int type_def_injected = 0;
        scan_and_inject(out, code, dtype, &injected_mask, &type_def_injected);
    }

    free(params);
    free(code);
    return out;
}
