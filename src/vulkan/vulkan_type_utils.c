/**
 * @file vulkan_type_utils.c
 * @brief Vulkan 类型辅助工具和动态 kxxx 函数注入
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
#include <vulkan/vulkan.h>
#include "ace.h"

/* 设备特性（由 vulkan_device.c 提供） */
typedef struct {
    int has_fp16;
    int has_int8;
    int has_int16;
    int has_16bit_storage;
    int has_8bit_storage;
} vk_device_features_t;
extern vk_device_features_t g_device_features;

/* ============================================================================
 * 设备能力查询
 * ============================================================================ */

static int vk_supports_native_kfunc(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:
        case ACE_DTYPE_INT32:
            return 1;
        case ACE_DTYPE_FLOAT64:
            return 1;  /* Vulkan 支持 float64 */
        case ACE_DTYPE_INT64:
            return 1;
        case ACE_DTYPE_FLOAT16:
            return g_device_features.has_fp16 && g_device_features.has_16bit_storage;
        case ACE_DTYPE_BFLOAT16:
            return 0;  /* BF16 总是需要模拟 */
        case ACE_DTYPE_INT8:
            return g_device_features.has_int8 && g_device_features.has_8bit_storage;
        case ACE_DTYPE_UINT8:
            return g_device_features.has_int8 && g_device_features.has_8bit_storage;
        case ACE_DTYPE_INT16:
            return g_device_features.has_int16 && g_device_features.has_16bit_storage;
        default:
            return 1;
    }
}

/* ============================================================================
 * 类型名称和扩展
 * ============================================================================ */

const char* vk_get_buffer_type_name(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:  return "float";
        case ACE_DTYPE_FLOAT64:  return "float64_t";
        case ACE_DTYPE_INT32:    return "int";
        case ACE_DTYPE_INT64:    return "int64_t";
        case ACE_DTYPE_FLOAT16:  return "float16_t";
        case ACE_DTYPE_BFLOAT16: return "uint16_t";
        case ACE_DTYPE_INT8:     return "int8_t";
        case ACE_DTYPE_UINT8:    return "uint8_t";
        case ACE_DTYPE_INT16:    return "int16_t";
        default:                 return "float";
    }
}

const char* vk_get_glsl_extension(ace_dtype_t dtype) {
    int use_native = vk_supports_native_kfunc(dtype);
    
    if (!use_native && (dtype == ACE_DTYPE_FLOAT16 || dtype == ACE_DTYPE_BFLOAT16 ||
                        dtype == ACE_DTYPE_INT8 || dtype == ACE_DTYPE_UINT8 ||
                        dtype == ACE_DTYPE_INT16)) {
        return "#extension GL_EXT_shader_explicit_arithmetic_types : require\n"
               "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n"
               "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
    }
    
    switch (dtype) {
        case ACE_DTYPE_FLOAT16:
            return "#extension GL_EXT_shader_16bit_storage : require\n"
                   "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
        case ACE_DTYPE_INT8:
            return "#extension GL_EXT_shader_8bit_storage : require\n"
                   "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
        case ACE_DTYPE_INT16:
            return "#extension GL_EXT_shader_16bit_storage : require\n"
                   "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
        default:
            return "";
    }
}

const char* vk_get_type_macros(ace_dtype_t dtype) {
    static char macros_buf[1024];
    const char* type_name = vk_get_buffer_type_name(dtype);
    snprintf(macros_buf, sizeof(macros_buf),
        "#define K_ZERO %s(0)\n"
        "#define K_ONE %s(1)\n"
        "#define K_NEG_ONE %s(-1)\n",
        type_name, type_name, type_name);
    return macros_buf;
}

/* ============================================================================
 * 类型定义和转换函数（用于非原生支持的类型）
 * ============================================================================ */

static const char* get_type_definition(ace_dtype_t dtype) {
    static char def_buf[4096];

    if (dtype == ACE_DTYPE_FLOAT16 && !vk_supports_native_kfunc(dtype)) {
        snprintf(def_buf, sizeof(def_buf),
            "/* FP16 模拟类型定义和转换函数 */\n"
            "float f16_to_f32(uint16_t x) {\n"
            "    uint sign = (x >> 15u) & 0x1u;\n"
            "    uint exp = (x >> 10u) & 0x1Fu;\n"
            "    uint man = x & 0x3FFu;\n"
            "    if (exp == 0u) return sign != 0u ? -0.0 : 0.0;\n"
            "    if (exp == 31u) return sign != 0u ? -1.0/0.0 : 1.0/0.0;\n"
            "    uint result = (sign << 31u) | ((exp + 112u) << 23u) | (man << 13u);\n"
            "    return uintBitsToFloat(result);\n"
            "}\n"
            "uint16_t f32_to_f16(float x) {\n"
            "    uint u = floatBitsToUint(x);\n"
            "    uint sign = (u >> 16u) & 0x8000u;\n"
            "    uint exp = ((u >> 23u) & 0xFFu) - 112u;\n"
            "    uint man = (u >> 13u) & 0x3FFu;\n"
            "    if ((u & 0x7FFFFFFFu) == 0u) return uint16_t(sign);\n"
            "    if (exp > 30u) return uint16_t(sign | 0x7C00u);\n"
            "    return uint16_t(sign | (exp << 10u) | man);\n"
            "}\n");
        return def_buf;
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        snprintf(def_buf, sizeof(def_buf),
            "/* BF16 模拟类型定义和转换函数 */\n"
            "float bf16_to_f32(uint16_t x) {\n"
            "    uint sign = (x >> 15) & 0x1u;\n"
            "    uint exp = (x >> 7) & 0xFFu;\n"
            "    uint man = x & 0x7Fu;\n"
            "    if (exp == 0u) return sign != 0u ? -0.0 : 0.0;\n"
            "    if (exp == 255u) return sign != 0u ? -1.0/0.0 : 1.0/0.0;\n"
            "    uint result = (sign << 31u) | (exp << 23u) | (man << 16u);\n"
            "    return uintBitsToFloat(result);\n"
            "}\n"
            "uint16_t f32_to_bf16(float x) {\n"
            "    uint u = floatBitsToUint(x);\n"
            "    uint sign = (u >> 16u) & 0x8000u;\n"
            "    uint exp = (u >> 23u) & 0xFFu;\n"
            "    uint man = (u >> 16u) & 0x7Fu;\n"
            "    if ((u & 0x7FFFFFFFu) == 0u) return uint16_t(sign);\n"
            "    if (exp == 255u) return uint16_t(sign | 0x7F80u);\n"
            "    return uint16_t(sign | (exp << 7u) | man);\n"
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
        const char* type_name = vk_get_buffer_type_name(dtype);
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

/**
 * @brief 扫描代码并注入所有找到的 kxxx 函数
 * @param out 输出缓冲区指针的指针（函数会修改它）
 * @param code 原始内核代码
 * @param dtype 数据类型
 * @param injected_mask 已注入函数掩码（输入/输出）
 */
static void scan_and_inject(char** out, const char* code, ace_dtype_t dtype, int* injected_mask) {
    int is_native = vk_supports_native_kfunc(dtype);

    char inject_buf[8192] = "";
    char* p = inject_buf;

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
        /* 查找类型定义结束的位置 */
        char* insert_pos = NULL;
        if (!is_native) {
            /* 非原生支持：在类型定义之后注入 */
            const char* marker = "*/\n";
            char* marker_pos = strstr(*out, marker);
            if (marker_pos) {
                insert_pos = marker_pos + strlen(marker);
            }
        }
        
        /* 如果找不到类型定义，就在开头注入 */
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
 * 代码翻译主函数
 * ============================================================================ */

char* vk_translate_to_glsl(const char* name, const char* src, ace_dtype_t dtype,
                           int* n_buffers, int* n_scalars) {
    (void)name;
    const char* body_start = strchr(src, '{');
    const char* body_end = strrchr(src, '}');
    if (!body_start || !body_end) {
        *n_buffers = 0;
        *n_scalars = 0;
        return strdup("#version 450\nlayout(local_size_x=256) in;\nvoid main(){}\n");
    }

    const char* buffer_type = vk_get_buffer_type_name(dtype);
    const char* extensions = vk_get_glsl_extension(dtype);
    const char* type_macros = vk_get_type_macros(dtype);

    /* 解析参数 */
    typedef struct {
        char name[64];
        int is_buffer;
    } param_info_t;

    param_info_t params[16];
    int n_params = 0;
    *n_buffers = 0;
    *n_scalars = 0;

    const char* p = strchr(src, '(');
    if (p) {
        p++;
        while (*p && *p != ')') {
            while (*p && (*p == ' ' || *p == '\t' || *p == '\n')) p++;
            if (*p == ')' || *p == '{') break;

            param_info_t* param = &params[n_params];
            char* dst = param->name;
            int is_ptr = 0;

            while (*p && *p != ',' && *p != ')' && n_params < 16) {
                if (*p == '*') {
                    is_ptr = 1;
                    p++;
                    continue;
                }
                if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '&') {
                    *dst++ = *p;
                }
                p++;
            }
            *dst = '\0';

            param->is_buffer = is_ptr;
            if (is_ptr) (*n_buffers)++;
            else (*n_scalars)++;
            n_params++;

            if (*p == ',') p++;
        }
    }

    /* 构建 push constants */
    char push_constants[1024] = "";
    char pc_access[1024] = "";

    if (*n_scalars > 0) {
        strcpy(push_constants, "layout(push_constant) uniform PC {\n");
        int scalar_idx = 0;
        for (int i = 0; i < n_params; i++) {
            if (!params[i].is_buffer) {
                char line[128];
                const char* scalar_type = (scalar_idx == 0) ? "int" : buffer_type;
                snprintf(line, sizeof(line), "  %s s%d;\n", scalar_type, scalar_idx);
                strcat(push_constants, line);
                char access[128];
                snprintf(access, sizeof(access), "#define %s pc.s%d\n", params[i].name, scalar_idx);
                strcat(pc_access, access);
                scalar_idx++;
            }
        }
        strcat(push_constants, "} pc;\n");
    }

    /* 构建 buffers */
    char buffers[2048] = "";
    int buf_idx = 0;
    for (int i = 0; i < n_params && buf_idx < *n_buffers; i++) {
        if (params[i].is_buffer) {
            char buf_decl[256];
            snprintf(buf_decl, sizeof(buf_decl),
                "layout(binding = %d, std430) buffer B%d { %s d%d[]; };\n",
                buf_idx, buf_idx, buffer_type, buf_idx);
            strcat(buffers, buf_decl);
            char def[128];
            snprintf(def, sizeof(def), "#define %s d%d\n", params[i].name, buf_idx);
            strcat(buffers, def);
            buf_idx++;
        }
    }

    /* 步骤 2: 扫描代码，查找有没有 kxxx 函数调用 */
    int has_k_functions = 0;
    const char* s = src;
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
    size_t len = 8192 + strlen(buffers) + strlen(push_constants) + strlen(pc_access) +
                 strlen(extensions) + strlen(type_macros) + (body_end - body_start);
    char* out = (char*)malloc(len);

    snprintf(out, len,
        "#version 450\n"
        "%s"
        "\nlayout(local_size_x = 256) in;\n"
        "%s"
        "%s"
        "%s"
        "%s"
        "#define GID int(gl_GlobalInvocationID.x)\n"
        "#define LID int(gl_LocalInvocationID.x)\n"
        "#define BSIZE 256\n"
        "#define BARRIER() barrier()\n"
        "void main() {\n"
        "%.*s"
        "}\n",
        extensions,
        buffers,
        push_constants,
        pc_access,
        type_macros,
        (int)(body_end - body_start - 1), body_start + 1
    );

    /* 步骤 4-6: 遍历所有 kxxx，注入对应的实现 */
    if (has_k_functions) {
        int injected_mask = 0;
        int is_native = vk_supports_native_kfunc(dtype);
        
        /* 非原生支持：先注入类型定义 */
        if (!is_native) {
            const char* type_def = get_type_definition(dtype);
            if (type_def && type_def[0]) {
                /* 找到 extensions 结束的位置 */
                char* insert_pos = strstr(out, extensions);
                if (insert_pos) {
                    insert_pos += strlen(extensions);
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
        scan_and_inject(&out, src, dtype, &injected_mask);
    }

    return out;
}
