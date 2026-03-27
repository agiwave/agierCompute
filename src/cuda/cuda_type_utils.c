/**
 * @file cuda_type_utils.c
 * @brief CUDA 类型辅助工具和动态 kxxx 函数注入
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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "ace.h"

/* ============================================================================
 * 设备能力查询
 * ============================================================================ */

/**
 * @brief 查询 CUDA 设备是否原生支持指定类型的 kxxx 运算
 * 
 * CUDA 设备对 FP16/BF16 的支持取决于计算能力 (Compute Capability):
 * - FP16 (half): Compute Capability >= 5.3 (Maxwell 之后) 支持原生 half 运算
 * - BF16: Compute Capability >= 8.0 (Ampere 之后) 支持原生 bfloat16 运算
 * - INT8: Compute Capability >= 6.1 (Volta 之后) 支持原生 int8 运算
 * - INT16: 大多数设备支持
 */
static int cuda_supports_native_kfunc(ace_dtype_t dtype) {
    /* 获取当前 CUDA 设备属性 */
    CUdevice device;
    CUresult result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        /* 如果无法获取设备信息，保守假设不支持 */
        return (dtype != ACE_DTYPE_FLOAT16 && dtype != ACE_DTYPE_BFLOAT16);
    }
    
    int compute_major = 0;
    int compute_minor = 0;
    
    /* 获取计算能力 */
    cuDeviceGetAttribute(&compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    
    int compute_capability = compute_major * 10 + compute_minor;
    
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:
        case ACE_DTYPE_FLOAT64:
        case ACE_DTYPE_INT32:
        case ACE_DTYPE_INT64:
            return 1;  /* 所有 CUDA 设备都原生支持 */
            
        case ACE_DTYPE_FLOAT16:
            /* FP16: Compute Capability >= 53 (5.3, Maxwell 之后) 支持原生 half 运算 */
            return (compute_capability >= 53);
            
        case ACE_DTYPE_BFLOAT16:
            /* BF16: Compute Capability >= 80 (8.0, Ampere 之后) 支持原生 bfloat16 运算 */
            return (compute_capability >= 80);
            
        case ACE_DTYPE_INT8:
        case ACE_DTYPE_UINT8:
            /* INT8: Compute Capability >= 61 (6.1, Pascal 之后) 支持原生 int8 运算 */
            return (compute_capability >= 61);
            
        case ACE_DTYPE_INT16:
            /* INT16: 大多数 CUDA 设备支持 */
            return (compute_capability >= 30);
            
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
    
    /* BF16 需要使用特殊的宏定义，避免类型转换歧义 */
    if (dtype == ACE_DTYPE_BFLOAT16) {
        snprintf(macros_buf, sizeof(macros_buf),
            "#define K_ZERO __float2bfloat16(0.0f)\n"
            "#define K_ONE __float2bfloat16(1.0f)\n"
            "#define K_NEG_ONE __float2bfloat16(-1.0f)\n");
    } else if (dtype == ACE_DTYPE_FLOAT16) {
        snprintf(macros_buf, sizeof(macros_buf),
            "#define K_ZERO __float2half(0.0f)\n"
            "#define K_ONE __float2half(1.0f)\n"
            "#define K_NEG_ONE __float2half(-1.0f)\n");
    } else {
        snprintf(macros_buf, sizeof(macros_buf),
            "#define K_ZERO (%s)0\n"
            "#define K_ONE (%s)1\n"
            "#define K_NEG_ONE (%s)-1\n",
            type_name, type_name, type_name);
    }
    return macros_buf;
}

/* ============================================================================
 * 类型定义和转换函数（用于非原生支持的类型）
 * ============================================================================ */

static const char* get_type_definition(ace_dtype_t dtype) {
    static char def_buf[4096];

    if (dtype == ACE_DTYPE_FLOAT16) {
        snprintf(def_buf, sizeof(def_buf),
            "/* FP16 类型定义和转换函数 */\n"
            "typedef __half_raw half_raw;\n"
            "__device__ inline half f32_to_f16(float f) { return __float2half(f); }\n"
            "__device__ inline float f16_to_f32(half h) { return __half2float(h); }\n");
        return def_buf;
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        snprintf(def_buf, sizeof(def_buf),
            "/* BF16 类型定义和转换函数 */\n"
            "typedef __nv_bfloat16_raw bfloat16_raw;\n"
            "__device__ inline __nv_bfloat16 f32_to_bf16(float f) { return __float2bfloat16(f); }\n"
            "__device__ inline float bf16_to_f32(__nv_bfloat16 h) { return __bfloat162float(h); }\n");
        return def_buf;
    }
    return "";
}

/* ============================================================================
 * 核心注入函数：注入单个 kxxx 函数
 * ============================================================================ */

static int inject_k_function_impl(char* buf, const char* func_name, ace_dtype_t dtype, int is_native) {
    const char* type_name = cuda_get_type_name(dtype);
    
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
        const char* type_name = cuda_get_type_name(dtype);
        
        /* 根据数据类型选择正确的转换函数 */
        const char* to_f32 = NULL;
        const char* from_f32 = NULL;
        
        if (dtype == ACE_DTYPE_FLOAT16) {
            to_f32 = "f16_to_f32";
            from_f32 = "f32_to_f16";
        } else if (dtype == ACE_DTYPE_BFLOAT16) {
            to_f32 = "bf16_to_f32";
            from_f32 = "f32_to_bf16";
        }
        /* INT8/INT16 等类型不需要转换函数，直接使用运算符 */
        
        if (to_f32 && from_f32) {
            /* FP16/BF16 需要转换函数 */
            if (strcmp(func_name, "kadd") == 0)
                return sprintf(buf, "__device__ inline %s kadd(%s a, %s b) { return %s(%s(a) + %s(b)); }\n",
                              type_name, type_name, type_name, from_f32, to_f32, to_f32);
            if (strcmp(func_name, "ksub") == 0)
                return sprintf(buf, "__device__ inline %s ksub(%s a, %s b) { return %s(%s(a) - %s(b)); }\n",
                              type_name, type_name, type_name, from_f32, to_f32, to_f32);
            if (strcmp(func_name, "kmul") == 0)
                return sprintf(buf, "__device__ inline %s kmul(%s a, %s b) { return %s(%s(a) * %s(b)); }\n",
                              type_name, type_name, type_name, from_f32, to_f32, to_f32);
            if (strcmp(func_name, "kdiv") == 0)
                return sprintf(buf, "__device__ inline %s kdiv(%s a, %s b) { return %s(%s(a) / %s(b)); }\n",
                              type_name, type_name, type_name, from_f32, to_f32, to_f32);
            if (strcmp(func_name, "klt") == 0)
                return sprintf(buf, "__device__ inline bool klt(%s a, %s b) { return %s(a) < %s(b); }\n",
                              type_name, type_name, to_f32, to_f32);
            if (strcmp(func_name, "kle") == 0)
                return sprintf(buf, "__device__ inline bool kle(%s a, %s b) { return %s(a) <= %s(b); }\n",
                              type_name, type_name, to_f32, to_f32);
            if (strcmp(func_name, "kgt") == 0)
                return sprintf(buf, "__device__ inline bool kgt(%s a, %s b) { return %s(a) > %s(b); }\n",
                              type_name, type_name, to_f32, to_f32);
            if (strcmp(func_name, "kge") == 0)
                return sprintf(buf, "__device__ inline bool kge(%s a, %s b) { return %s(a) >= %s(b); }\n",
                              type_name, type_name, to_f32, to_f32);
            if (strcmp(func_name, "keq") == 0)
                return sprintf(buf, "__device__ inline bool keq(%s a, %s b) { return %s(a) == %s(b); }\n",
                              type_name, type_name, to_f32, to_f32);
            if (strcmp(func_name, "kne") == 0)
                return sprintf(buf, "__device__ inline bool kne(%s a, %s b) { return %s(a) != %s(b); }\n",
                              type_name, type_name, to_f32, to_f32);
        } else {
            /* INT8/INT16 等类型直接使用运算符 */
            if (strcmp(func_name, "kadd") == 0)
                return sprintf(buf, "__device__ inline %s kadd(%s a, %s b) { return (a) + (b); }\n",
                              type_name, type_name, type_name);
            if (strcmp(func_name, "ksub") == 0)
                return sprintf(buf, "__device__ inline %s ksub(%s a, %s b) { return (a) - (b); }\n",
                              type_name, type_name, type_name);
            if (strcmp(func_name, "kmul") == 0)
                return sprintf(buf, "__device__ inline %s kmul(%s a, %s b) { return (a) * (b); }\n",
                              type_name, type_name, type_name);
            if (strcmp(func_name, "kdiv") == 0)
                return sprintf(buf, "__device__ inline %s kdiv(%s a, %s b) { return (a) / (b); }\n",
                              type_name, type_name, type_name);
            if (strcmp(func_name, "klt") == 0)
                return sprintf(buf, "__device__ inline bool klt(%s a, %s b) { return (a) < (b); }\n",
                              type_name, type_name);
            if (strcmp(func_name, "kle") == 0)
                return sprintf(buf, "__device__ inline bool kle(%s a, %s b) { return (a) <= (b); }\n",
                              type_name, type_name);
            if (strcmp(func_name, "kgt") == 0)
                return sprintf(buf, "__device__ inline bool kgt(%s a, %s b) { return (a) > (b); }\n",
                              type_name, type_name);
            if (strcmp(func_name, "kge") == 0)
                return sprintf(buf, "__device__ inline bool kge(%s a, %s b) { return (a) >= (b); }\n",
                              type_name, type_name);
            if (strcmp(func_name, "keq") == 0)
                return sprintf(buf, "__device__ inline bool keq(%s a, %s b) { return (a) == (b); }\n",
                              type_name, type_name);
            if (strcmp(func_name, "kne") == 0)
                return sprintf(buf, "__device__ inline bool kne(%s a, %s b) { return (a) != (b); }\n",
                              type_name, type_name);
        }
    }
    return 0;
}

/* ============================================================================
 * 扫描代码并注入所有找到的 kxxx 函数
 * ============================================================================ */

/**
 * @brief 扫描代码并注入所有找到的 kxxx 函数
 * 流程:
 * 1. 扫描找到所有 kxxx( 模式
 * 2. 对每个找到的 kxxx:
 *    - 检查是否已注入（使用 injected_mask 去重）
 *    - 注入 kxxx 函数实现（根据 is_native 决定注入宏还是函数）
 * @param out 输出缓冲区指针的指针（函数会修改它）
 */
static void scan_and_inject(char** out, const char* code, ace_dtype_t dtype, int* injected_mask) {
    int is_native = cuda_supports_native_kfunc(dtype);

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

    /* 将注入的函数插入到输出代码中 - 在类型定义之后，内核函数之前 */
    if (p > inject_buf) {
        /* 查找注入位置 */
        char* insert_pos = NULL;
        
        if (!is_native) {
            /* 非原生支持：在类型定义之后注入 */
            /* 找到类型转换函数定义的结束位置 */
            const char* end_marker = NULL;
            if (dtype == ACE_DTYPE_FLOAT16) {
                end_marker = "f16_to_f32(half h) { return __half2float(h); }\n";
            } else if (dtype == ACE_DTYPE_BFLOAT16) {
                end_marker = "bf16_to_f32(__nv_bfloat16 h) { return __bfloat162float(h); }\n";
            }
            
            if (end_marker) {
                char* func_end = strstr(*out, end_marker);
                if (func_end) {
                    insert_pos = func_end + strlen(end_marker);
                }
            }
        }

        /* 如果找不到类型定义，就在 extern 之前注入 */
        if (!insert_pos) {
            insert_pos = strstr(*out, "extern \"C\"");
        }
        /* 如果还找不到，就在开头注入 */
        if (!insert_pos) insert_pos = *out;

        /* 在 insert_pos 位置注入 kxxx 函数 */
        size_t prefix_len = insert_pos - *out;
        size_t new_len = strlen(*out) + strlen(inject_buf) + 10;
        char* temp = (char*)malloc(new_len);
        if (temp) {
            strncpy(temp, *out, prefix_len);
            temp[prefix_len] = '\0';
            strcat(temp, inject_buf);
            strcat(temp, "\n");
            strcat(temp, insert_pos);
            free(*out);
            *out = temp;
        }
    }
}

/* ============================================================================
 * 代码翻译主函数 - 完整流程
 * ============================================================================ */

char* cuda_translate_code(const char* name, const char* src, ace_dtype_t dtype) {
    const char* type_name = cuda_get_type_name(dtype);
    const char* type_headers = cuda_get_type_headers(dtype);
    const char* type_macros = cuda_get_type_macros(dtype);

    /* 步骤 1: 替换 T 为实际类型 */
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
                       strlen(type_headers) + strlen(type_macros);
    char* out = (char*)malloc(total_len);

    snprintf(out, total_len,
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
        type_macros,
        name, params,
        (int)body_len, body_start + 1
    );

    /* 步骤 4-6: 遍历所有 kxxx，注入对应的实现，然后编译 */
    if (has_k_functions) {
        int injected_mask = 0;
        int is_native = cuda_supports_native_kfunc(dtype);
        
        /* 非原生支持：先注入类型定义（在头文件之后） */
        if (!is_native) {
            const char* type_def = get_type_definition(dtype);
            if (type_def && type_def[0]) {
                /* 找到 type_headers 结束的位置 */
                char* insert_pos = strstr(out, type_headers);
                if (insert_pos) {
                    insert_pos += strlen(type_headers);
                    /* 在 insert_pos 位置插入类型定义 */
                    size_t prefix_len = insert_pos - out;
                    size_t new_len = strlen(out) + strlen(type_def) + 10;
                    char* temp = (char*)malloc(new_len);
                    if (temp) {
                        strncpy(temp, out, prefix_len);
                        temp[prefix_len] = '\0';
                        strcat(temp, type_def);
                        strcat(temp, "\n");
                        strcat(temp, insert_pos);
                        free(out);
                        out = temp;
                    }
                }
            }
        }

        /* 扫描并注入所有找到的 kxxx 函数（在类型定义之后） */
        scan_and_inject(&out, code, dtype, &injected_mask);
    }

    free(params);
    free(code);
    return out;
}
