/**
 * @file cuda_type_utils.c
 * @brief CUDA 类型辅助工具和动态 kxxx 函数注入
 * 
 * 动态注入机制:
 * 1. 扫描用户内核代码，找到所有 kxxx( 模式
 * 2. 对每个找到的 kxxx，调用 inject_k_function() 注入
 * 3. inject_k_function 根据 func_name + dtype + is_native 决定注入什么
 * 4. 使用 injected_mask 避免重复注入
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
 * 设备能力查询（实际应由设备代码提供，这里简化实现）
 * ============================================================================ */

/* 查询类型是否原生支持 kxxx 运算 */
static int cuda_supports_native_kfunc(ace_dtype_t dtype) {
    /* 实际应该查询设备属性，这里根据 CUDA 能力简化判断 */
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:
        case ACE_DTYPE_FLOAT64:
        case ACE_DTYPE_INT32:
        case ACE_DTYPE_INT64:
        case ACE_DTYPE_INT8:
        case ACE_DTYPE_UINT8:
        case ACE_DTYPE_INT16:
            return 1;  /* 原生支持 */
        case ACE_DTYPE_FLOAT16:
        case ACE_DTYPE_BFLOAT16:
            return 0;  /* 需要模拟 */
        default:
            return 1;
    }
}

/* ============================================================================
 * 类型名称和头文件
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

const char* cuda_get_type_macros(ace_dtype_t dtype) {
    static char macros_buf[1024];
    const char* type_name = cuda_get_type_name(dtype);
    snprintf(macros_buf, sizeof(macros_buf),
        "#define K_ZERO (%s)0\n"
        "#define K_ONE (%s)1\n"
        "#define K_NEG_ONE (%s)-1\n",
        type_name, type_name, type_name);
    return macros_buf;
}

/* ============================================================================
 * 类型转换函数（仅用于非原生支持的类型）
 * ============================================================================ */

static const char* get_type_converters(ace_dtype_t dtype) {
    static char conv_buf[4096];

    if (dtype == ACE_DTYPE_FLOAT16) {
        snprintf(conv_buf, sizeof(conv_buf),
            "typedef __half_raw half_raw;\n"
            "__device__ inline half f32_to_f16(float f) { return __float2half(f); }\n"
            "__device__ inline float f16_to_f32(half h) { return __half2float(h); }\n"
            "#define K_ZERO __float2half(0.0f)\n"
            "#define K_ONE __float2half(1.0f)\n"
            "#define K_NEG_ONE __float2half(-1.0f)\n");
        return conv_buf;
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        snprintf(conv_buf, sizeof(conv_buf),
            "typedef __nv_bfloat16_raw bfloat16_raw;\n"
            "__device__ inline __nv_bfloat16 f32_to_bf16(float f) { return __float2bfloat16(f); }\n"
            "__device__ inline float bf16_to_f32(__nv_bfloat16 h) { return __bfloat162float(h); }\n"
            "#define K_ZERO __float2bfloat16(0.0f)\n"
            "#define K_ONE __float2bfloat16(1.0f)\n"
            "#define K_NEG_ONE __float2bfloat16(-1.0f)\n");
        return conv_buf;
    }
    return "";
}

/* ============================================================================
 * 核心注入函数：根据 func_name + dtype + is_native 决定注入什么
 * ============================================================================ */

/**
 * @brief 注入单个 kxxx 函数
 * @param buf 输出缓冲区
 * @param func_name 函数名（如 "kadd", "ksub" 等）
 * @param dtype 数据类型
 * @param is_native 是否原生支持（1=原生，0=模拟）
 * @return 注入的代码长度
 */
static int inject_k_function(char* buf, const char* func_name, ace_dtype_t dtype, int is_native) {
    const char* type_name = cuda_get_type_name(dtype);
    
    if (is_native) {
        /* 原生支持：注入宏定义 */
        if (strcmp(func_name, "kadd") == 0) {
            return sprintf(buf, "#define kadd(a, b) ((a) + (b))\n");
        }
        if (strcmp(func_name, "ksub") == 0) {
            return sprintf(buf, "#define ksub(a, b) ((a) - (b))\n");
        }
        if (strcmp(func_name, "kmul") == 0) {
            return sprintf(buf, "#define kmul(a, b) ((a) * (b))\n");
        }
        if (strcmp(func_name, "kdiv") == 0) {
            return sprintf(buf, "#define kdiv(a, b) ((a) / (b))\n");
        }
        if (strcmp(func_name, "klt") == 0) {
            return sprintf(buf, "#define klt(a, b) ((a) < (b))\n");
        }
        if (strcmp(func_name, "kle") == 0) {
            return sprintf(buf, "#define kle(a, b) ((a) <= (b))\n");
        }
        if (strcmp(func_name, "kgt") == 0) {
            return sprintf(buf, "#define kgt(a, b) ((a) > (b))\n");
        }
        if (strcmp(func_name, "kge") == 0) {
            return sprintf(buf, "#define kge(a, b) ((a) >= (b))\n");
        }
        if (strcmp(func_name, "keq") == 0) {
            return sprintf(buf, "#define keq(a, b) ((a) == (b))\n");
        }
        if (strcmp(func_name, "kne") == 0) {
            return sprintf(buf, "#define kne(a, b) ((a) != (b))\n");
        }
    } else {
        /* 非原生支持：注入模拟函数 */
        const char* to_f32 = (dtype == ACE_DTYPE_FLOAT16) ? "f16_to_f32" : "bf16_to_f32";
        const char* from_f32 = (dtype == ACE_DTYPE_FLOAT16) ? "f32_to_f16" : "f32_to_bf16";
        
        if (strcmp(func_name, "kadd") == 0) {
            return sprintf(buf,
                "__device__ inline %s kadd(%s a, %s b) { "
                "  return %s(%s(a) + %s(b)); "
                "}\n",
                type_name, type_name, type_name, from_f32, to_f32, to_f32);
        }
        if (strcmp(func_name, "ksub") == 0) {
            return sprintf(buf,
                "__device__ inline %s ksub(%s a, %s b) { "
                "  return %s(%s(a) - %s(b)); "
                "}\n",
                type_name, type_name, type_name, from_f32, to_f32, to_f32);
        }
        if (strcmp(func_name, "kmul") == 0) {
            return sprintf(buf,
                "__device__ inline %s kmul(%s a, %s b) { "
                "  return %s(%s(a) * %s(b)); "
                "}\n",
                type_name, type_name, type_name, from_f32, to_f32, to_f32);
        }
        if (strcmp(func_name, "kdiv") == 0) {
            return sprintf(buf,
                "__device__ inline %s kdiv(%s a, %s b) { "
                "  return %s(%s(a) / %s(b)); "
                "}\n",
                type_name, type_name, type_name, from_f32, to_f32, to_f32);
        }
        if (strcmp(func_name, "klt") == 0) {
            return sprintf(buf,
                "__device__ inline bool klt(%s a, %s b) { "
                "  return %s(a) < %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        }
        if (strcmp(func_name, "kle") == 0) {
            return sprintf(buf,
                "__device__ inline bool kle(%s a, %s b) { "
                "  return %s(a) <= %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        }
        if (strcmp(func_name, "kgt") == 0) {
            return sprintf(buf,
                "__device__ inline bool kgt(%s a, %s b) { "
                "  return %s(a) > %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        }
        if (strcmp(func_name, "kge") == 0) {
            return sprintf(buf,
                "__device__ inline bool kge(%s a, %s b) { "
                "  return %s(a) >= %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        }
        if (strcmp(func_name, "keq") == 0) {
            return sprintf(buf,
                "__device__ inline bool keq(%s a, %s b) { "
                "  return %s(a) == %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        }
        if (strcmp(func_name, "kne") == 0) {
            return sprintf(buf,
                "__device__ inline bool kne(%s a, %s b) { "
                "  return %s(a) != %s(b); "
                "}\n",
                type_name, type_name, to_f32, to_f32);
        }
    }
    return 0;
}

/* ============================================================================
 * 扫描代码并注入所有找到的 kxxx 函数
 * ============================================================================ */

/**
 * @brief 扫描代码并注入所有找到的 kxxx 函数
 * @param out 输出缓冲区（注入到当前位置）
 * @param code 原始内核代码
 * @param dtype 数据类型
 * @param injected_mask 已注入函数掩码（输入/输出）
 */
static void scan_and_inject(char* out, const char* code, ace_dtype_t dtype, int* injected_mask) {
    /* 查询设备是否原生支持该类型的 kxxx 运算 */
    int is_native = cuda_supports_native_kfunc(dtype);
    
    /* 检查是否已经全部注入过了 */
    const int ALL_FUNCTIONS = 0x3FF;  /* 10 个函数 */
    if (*injected_mask == ALL_FUNCTIONS) {
        return;
    }

    char inject_buf[8192] = "";
    char* p = inject_buf;
    char* end = inject_buf + sizeof(inject_buf) - 1;

    /* 扫描代码，查找所有 kxxx( 模式 */
    const char* s = code;
    while (*s) {
        /* 查找 k 开头的标识符 */
        if (islower(*s) && s[1] && islower(s[1]) && s[2] && islower(s[2])) {
            const char* start = s;
            while (islower(*s)) s++;

            /* 检查后面是否是 '(' */
            if (*s == '(') {
                /* 提取函数名 */
                char func_name[16];
                size_t len = s - start;
                if (len < sizeof(func_name)) {
                    strncpy(func_name, start, len);
                    func_name[len] = '\0';

                    /* 检查是否是 kxxx 函数 */
                    if (func_name[0] == 'k' && func_name[1] && func_name[2]) {
                        /* 计算函数 ID 用于去重 */
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

                        /* 只注入还未注入的函数 */
                        if (func_id >= 0 && func_id < 10 && !(*injected_mask & (1 << func_id))) {
                            /* 核心：调用注入函数，根据 func_name + dtype + is_native 决定注入什么 */
                            int inj_len = inject_k_function(p, func_name, dtype, is_native);
                            if (inj_len > 0) {
                                p += inj_len;
                                *injected_mask |= (1 << func_id);  /* 标记为已注入 */
                            }
                        }
                    }
                }
            }
        } else {
            s++;
        }
    }

    /* 将注入的函数插入到输出代码中 */
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

char* cuda_translate_code(const char* name, const char* src, ace_dtype_t dtype) {
    const char* type_name = cuda_get_type_name(dtype);
    const char* type_headers = cuda_get_type_headers(dtype);
    const char* type_macros = cuda_get_type_macros(dtype);
    const char* converters = get_type_converters(dtype);

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

    /* 扫描代码，查找有没有 kxxx 函数调用 */
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

    /* 判断是否需要转换函数（只有非原生支持且使用了 kxxx 才需要） */
    int need_converters = !cuda_supports_native_kfunc(dtype) && has_k_functions;

    size_t conv_len = need_converters ? strlen(converters) : 0;
    size_t total_len = strlen(name) + params_len + body_len + 2048 +
                       strlen(type_headers) + strlen(type_macros) + conv_len;
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
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
        type_macros,
        name, params,
        (int)body_len, body_start + 1
    );

    /* 扫描代码并注入所有找到的 kxxx 函数 */
    if (has_k_functions) {
        int injected_mask = 0;  /* 初始化为未注入任何函数 */
        /* 找到 converters 结束的位置，或者在 type_macros 之后 */
        char* inject_pos = NULL;
        if (need_converters) {
            inject_pos = strstr(out, converters);
            if (inject_pos) inject_pos += strlen(converters);
        } else {
            /* 原生类型：在 type_macros 之后注入 */
            inject_pos = strstr(out, type_macros);
            if (inject_pos) inject_pos += strlen(type_macros);
        }
        /* 如果找不到位置，就在开头注入 */
        if (!inject_pos) inject_pos = out;
        /* 在当前位置注入 */
        scan_and_inject(inject_pos, code, dtype, &injected_mask);
    }

    free(params);
    free(code);
    return out;
}
