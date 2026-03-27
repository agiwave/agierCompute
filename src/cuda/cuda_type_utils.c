/**
 * @file cuda_type_utils.c
 * @brief CUDA 类型辅助工具和动态 kxxx 函数注入
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "ace.h"

/* ============================================================================
 * 类型名称映射
 * ============================================================================ */

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

/* ============================================================================
 * 类型头文件
 * ============================================================================ */

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

/* ============================================================================
 * 类型宏定义
 * ============================================================================ */

const char* cuda_get_type_macros(ace_dtype_t dtype) {
    (void)dtype;
    return "";
}

/* ============================================================================
 * 内核函数宏（基础类型直接展开）
 * ============================================================================ */

const char* cuda_get_kernel_macros(ace_dtype_t dtype) {
    static char macros_buf[2048];
    
    if (dtype == ACE_DTYPE_FLOAT16) {
        snprintf(macros_buf, sizeof(macros_buf),
            "#define K_ZERO half(0)\n"
            "#define K_ONE half(1)\n"
            "#define K_NEG_ONE half(-1)\n");
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        snprintf(macros_buf, sizeof(macros_buf),
            "#define K_ZERO __float2bfloat16(0.0f)\n"
            "#define K_ONE __float2bfloat16(1.0f)\n"
            "#define K_NEG_ONE __float2bfloat16(-1.0f)\n");
    } else {
        const char* type_name = cuda_get_type_name(dtype);
        snprintf(macros_buf, sizeof(macros_buf),
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
 * 类型转换函数（只包含转换，不包含运算）
 * ============================================================================ */

static const char* get_type_converters(ace_dtype_t dtype) {
    static char conv_buf[2048];

    if (dtype == ACE_DTYPE_FLOAT16) {
        snprintf(conv_buf, sizeof(conv_buf),
            "typedef __half_raw half_raw;\n"
            "__device__ inline half f32_to_f16(float f) { return __float2half(f); }\n"
            "__device__ inline float f16_to_f32(half h) { return __half2float(h); }\n");
        return conv_buf;
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        snprintf(conv_buf, sizeof(conv_buf),
            "typedef __nv_bfloat16_raw bfloat16_raw;\n"
            "__device__ inline __nv_bfloat16 f32_to_bf16(float f) { return __float2bfloat16(f); }\n"
            "__device__ inline float bf16_to_f32(__nv_bfloat16 h) { return __bfloat162float(h); }\n");
        return conv_buf;
    }
    return "";
}

/* ============================================================================
 * 动态 kxxx 函数注入 - 使用正则表达式提取并注入
 * ============================================================================ */

/* 支持的 kxxx 函数列表 */
typedef enum {
    K_FUNC_ADD,
    K_FUNC_SUB,
    K_FUNC_MUL,
    K_FUNC_DIV,
    K_FUNC_LT,
    K_FUNC_LE,
    K_FUNC_GT,
    K_FUNC_GE,
    K_FUNC_EQ,
    K_FUNC_NE,
    K_FUNC_COUNT
} k_func_id_t;

static const char* k_func_names[] = {
    "kadd", "ksub", "kmul", "kdiv",
    "klt", "kle", "kgt", "kge", "keq", "kne"
};

/* 从代码中提取所有 kxxx( 模式的函数调用 */
static int extract_k_functions(const char* code, int* found_funcs) {
    int count = 0;
    const char* p = code;
    
    /* 初始化 */
    for (int i = 0; i < K_FUNC_COUNT; i++) {
        found_funcs[i] = 0;
    }
    
    /* 使用类似正则的方式匹配 kxxx( 模式 */
    while (*p) {
        /* 查找 'k' 开头 */
        if (*p == 'k') {
            /* 检查是否是 kxxx( 模式 */
            if (p[1] == 'a' && p[2] == 'd' && p[3] == 'd' && p[4] == '(') {
                found_funcs[K_FUNC_ADD] = 1;
                p += 5;
                continue;
            }
            if (p[1] == 's' && p[2] == 'u' && p[3] == 'b' && p[4] == '(') {
                found_funcs[K_FUNC_SUB] = 1;
                p += 5;
                continue;
            }
            if (p[1] == 'm' && p[2] == 'u' && p[3] == 'l' && p[4] == '(') {
                found_funcs[K_FUNC_MUL] = 1;
                p += 5;
                continue;
            }
            if (p[1] == 'd' && p[2] == 'i' && p[3] == 'v' && p[4] == '(') {
                found_funcs[K_FUNC_DIV] = 1;
                p += 5;
                continue;
            }
            if (p[1] == 'l' && p[2] == 't' && p[3] == '(') {
                found_funcs[K_FUNC_LT] = 1;
                p += 4;
                continue;
            }
            if (p[1] == 'l' && p[2] == 'e' && p[3] == '(') {
                found_funcs[K_FUNC_LE] = 1;
                p += 4;
                continue;
            }
            if (p[1] == 'g' && p[2] == 't' && p[3] == '(') {
                found_funcs[K_FUNC_GT] = 1;
                p += 4;
                continue;
            }
            if (p[1] == 'g' && p[2] == 'e' && p[3] == '(') {
                found_funcs[K_FUNC_GE] = 1;
                p += 4;
                continue;
            }
            if (p[1] == 'e' && p[2] == 'q' && p[3] == '(') {
                found_funcs[K_FUNC_EQ] = 1;
                p += 4;
                continue;
            }
            if (p[1] == 'n' && p[2] == 'e' && p[3] == '(') {
                found_funcs[K_FUNC_NE] = 1;
                p += 4;
                continue;
            }
        }
        p++;
    }
    
    /* 计算找到的函数数量 */
    for (int i = 0; i < K_FUNC_COUNT; i++) {
        if (found_funcs[i]) count++;
    }
    
    return count;
}

/* 生成单个 kxxx 函数的实现 */
static int generate_k_function(char* buf, size_t buf_size, k_func_id_t func_id, 
                                const char* type_name, const char* to_f32, 
                                const char* from_f32) {
    (void)func_id;  /* 用于 switch 判断 */
    
    switch (func_id) {
        case K_FUNC_ADD:
            return snprintf(buf, buf_size,
                "__device__ inline %s kadd(%s a, %s b) { "
                "  return %s(%s(a) + %s(b)); "
                "}\n",
                type_name, type_name, type_name, from_f32, to_f32, to_f32);
        case K_FUNC_SUB:
            return snprintf(buf, buf_size,
                "__device__ inline %s ksub(%s a, %s b) { "
                "  return %s(%s(a) - %s(b)); "
                "}\n",
                type_name, type_name, type_name, from_f32, to_f32, to_f32);
        case K_FUNC_MUL:
            return snprintf(buf, buf_size,
                "__device__ inline %s kmul(%s a, %s b) { "
                "  return %s(%s(a) * %s(b)); "
                "}\n",
                type_name, type_name, type_name, from_f32, to_f32, to_f32);
        case K_FUNC_DIV:
            return snprintf(buf, buf_size,
                "__device__ inline %s kdiv(%s a, %s b) { "
                "  return %s(%s(a) / %s(b)); "
                "}\n",
                type_name, type_name, type_name, from_f32, to_f32, to_f32);
        case K_FUNC_LT:
            return snprintf(buf, buf_size,
                "__device__ inline bool klt(%s a, %s b) { "
                "  return %s(a) < %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        case K_FUNC_LE:
            return snprintf(buf, buf_size,
                "__device__ inline bool kle(%s a, %s b) { "
                "  return %s(a) <= %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        case K_FUNC_GT:
            return snprintf(buf, buf_size,
                "__device__ inline bool kgt(%s a, %s b) { "
                "  return %s(a) > %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        case K_FUNC_GE:
            return snprintf(buf, buf_size,
                "__device__ inline bool kge(%s a, %s b) { "
                "  return %s(a) >= %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        case K_FUNC_EQ:
            return snprintf(buf, buf_size,
                "__device__ inline bool keq(%s a, %s b) { "
                "  return %s(a) == %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        case K_FUNC_NE:
            return snprintf(buf, buf_size,
                "__device__ inline bool kne(%s a, %s b) { "
                "  return %s(a) != %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        default:
            return 0;
    }
}

/* 动态注入实际使用到的 kxxx 函数 */
static void inject_used_k_functions(char* out, const char* code, const char* converters, ace_dtype_t dtype) {
    if (dtype != ACE_DTYPE_FLOAT16 && dtype != ACE_DTYPE_BFLOAT16) {
        return;  /* 基础类型不需要注入 */
    }

    int found_funcs[K_FUNC_COUNT];
    int count = extract_k_functions(code, found_funcs);
    
    if (count == 0) {
        return;  /* 没有使用任何 kxxx 函数 */
    }

    char inject_buf[8192] = "";
    char* p = inject_buf;
    char* end = inject_buf + sizeof(inject_buf) - 1;

    const char* type_name = (dtype == ACE_DTYPE_FLOAT16) ? "half" : "__nv_bfloat16";
    const char* to_f32 = (dtype == ACE_DTYPE_FLOAT16) ? "f16_to_f32" : "bf16_to_f32";
    const char* from_f32 = (dtype == ACE_DTYPE_FLOAT16) ? "f32_to_f16" : "f32_to_bf16";

    /* 遍历所有找到的函数，生成对应的实现 */
    for (int i = 0; i < K_FUNC_COUNT; i++) {
        if (found_funcs[i]) {
            int len = generate_k_function(p, end - p, (k_func_id_t)i, 
                                          type_name, to_f32, from_f32);
            p += len;
        }
    }

    /* 将注入的函数插入到输出代码中 - 在 converters 之后 */
    if (inject_buf[0] != '\0') {
        /* 找到 converters 结束的位置，插入到那里 */
        char* inject_pos = strstr(out, converters);
        if (inject_pos) {
            inject_pos += strlen(converters);
            /* 创建新缓冲区 */
            char* temp = (char*)malloc(strlen(out) + strlen(inject_buf) + 10);
            if (temp) {
                size_t prefix_len = inject_pos - out;
                strncpy(temp, out, prefix_len);
                temp[prefix_len] = '\0';
                strcat(temp, "\n");
                strcat(temp, inject_buf);
                strcat(temp, inject_pos);
                strcpy(out, temp);
                free(temp);
            }
        } else {
            /* 如果没有 converters，插入到开头 */
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
}

/* ============================================================================
 * 代码翻译主函数
 * ============================================================================ */

char* cuda_translate_code(const char* name, const char* src, ace_dtype_t dtype) {
    const char* type_name = cuda_get_type_name(dtype);
    const char* type_headers = cuda_get_type_headers(dtype);
    const char* type_macros = cuda_get_type_macros(dtype);
    const char* converters = get_type_converters(dtype);
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

    /* 检测是否需要转换函数 */
    int need_converters = 0;
    if (dtype == ACE_DTYPE_FLOAT16 || dtype == ACE_DTYPE_BFLOAT16) {
        int found_funcs[K_FUNC_COUNT];
        if (extract_k_functions(code, found_funcs) > 0) {
            need_converters = 1;
        }
    }

    size_t conv_len = need_converters ? strlen(converters) : 0;
    size_t total_len = strlen(name) + params_len + body_len + 2048 +
                       strlen(type_headers) + strlen(type_macros) + 
                       strlen(kernel_macros) + conv_len;
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
        need_converters ? converters : "",
        kernel_macros,
        type_macros,
        name, params,
        (int)body_len, body_start + 1
    );

    /* 动态注入实际使用到的 kxxx 函数 */
    if (need_converters) {
        inject_used_k_functions(out, code, converters, dtype);
    }

    free(params);
    free(code);
    return out;
}
