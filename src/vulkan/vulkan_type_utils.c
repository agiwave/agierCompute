/**
 * @file vulkan_type_utils.c
 * @brief Vulkan 类型辅助工具和动态 kxxx 函数注入
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "ace.h"

/* ============================================================================
 * 设备特性检测（由 vulkan_device.c 提供）
 * ============================================================================ */

typedef struct {
    int has_fp16;
    int has_int8;
    int has_int16;
    int has_16bit_storage;
    int has_8bit_storage;
} vk_device_features_t;
extern vk_device_features_t g_device_features;

static int vk_supports_native_storage(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT16:
            return g_device_features.has_fp16 && g_device_features.has_16bit_storage;
        case ACE_DTYPE_BFLOAT16:
            return 0;  /* BF16 总是模拟 */
        case ACE_DTYPE_INT8:
            return g_device_features.has_int8 && g_device_features.has_8bit_storage;
        case ACE_DTYPE_INT16:
            return g_device_features.has_int16 && g_device_features.has_16bit_storage;
        default:
            return 1;
    }
}

/* ============================================================================
 * 类型名称映射
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

const char* vk_get_compute_type_name(ace_dtype_t dtype) {
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

/* ============================================================================
 * GLSL 扩展声明
 * ============================================================================ */

const char* vk_get_glsl_extension(ace_dtype_t dtype) {
    int use_native = vk_supports_native_storage(dtype);
    
    if (!use_native) {
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

/* ============================================================================
 * 只包含类型转换函数，运算函数动态生成
 * ============================================================================ */

static const char* get_type_converters(ace_dtype_t dtype) {
    static char conv_buf[4096];
    
    if (dtype == ACE_DTYPE_FLOAT16) {
        snprintf(conv_buf, sizeof(conv_buf),
            "/* FP16 类型转换函数 */\n"
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
        return conv_buf;
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        snprintf(conv_buf, sizeof(conv_buf),
            "/* BF16 类型转换函数 */\n"
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
        return conv_buf;
    }
    return "";
}

/* ============================================================================
 * 动态 kxxx 函数注入
 * ============================================================================ */

static void inject_used_k_functions(char* out, size_t out_size, const char* code, ace_dtype_t dtype) {
    if (dtype != ACE_DTYPE_FLOAT16 && dtype != ACE_DTYPE_BFLOAT16) {
        return;
    }

    char inject_buf[8192] = "";
    char* p = inject_buf;
    char* end = inject_buf + sizeof(inject_buf) - 1;

    const char* type_name = "uint16_t";
    const char* to_f32 = (dtype == ACE_DTYPE_FLOAT16) ? "f16_to_f32" : "bf16_to_f32";
    const char* from_f32 = (dtype == ACE_DTYPE_FLOAT16) ? "f32_to_f16" : "f32_to_bf16";

    /* 检测并注入 kadd */
    if (strstr(code, "kadd")) {
        int len = snprintf(p, end - p,
            "uint16_t kadd(uint16_t a, uint16_t b) { "
            "  return %s(%s(a) + %s(b)); "
            "}\n",
            from_f32, to_f32, to_f32);
        p += len;
    }

    /* 检测并注入 ksub */
    if (strstr(code, "ksub")) {
        int len = snprintf(p, end - p,
            "uint16_t ksub(uint16_t a, uint16_t b) { "
            "  return %s(%s(a) - %s(b)); "
            "}\n",
            from_f32, to_f32, to_f32);
        p += len;
    }

    /* 检测并注入 kmul */
    if (strstr(code, "kmul")) {
        int len = snprintf(p, end - p,
            "uint16_t kmul(uint16_t a, uint16_t b) { "
            "  return %s(%s(a) * %s(b)); "
            "}\n",
            from_f32, to_f32, to_f32);
        p += len;
    }

    /* 检测并注入 kdiv */
    if (strstr(code, "kdiv")) {
        int len = snprintf(p, end - p,
            "uint16_t kdiv(uint16_t a, uint16_t b) { "
            "  return %s(%s(a) / %s(b)); "
            "}\n",
            from_f32, to_f32, to_f32);
        p += len;
    }

    /* 检测并注入比较函数 */
    if (strstr(code, "klt")) {
        int len = snprintf(p, end - p,
            "bool klt(uint16_t a, uint16_t b) { "
            "  return %s(a) < %s(b); "
            "}\n",
            to_f32, to_f32);
        p += len;
    }
    if (strstr(code, "kle")) {
        int len = snprintf(p, end - p,
            "bool kle(uint16_t a, uint16_t b) { "
            "  return %s(a) <= %s(b); "
            "}\n",
            to_f32, to_f32);
        p += len;
    }
    if (strstr(code, "kgt")) {
        int len = snprintf(p, end - p,
            "bool kgt(uint16_t a, uint16_t b) { "
            "  return %s(a) > %s(b); "
            "}\n",
            to_f32, to_f32);
        p += len;
    }
    if (strstr(code, "kge")) {
        int len = snprintf(p, end - p,
            "bool kge(uint16_t a, uint16_t b) { "
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

static const char* vk_get_kernel_macros(ace_dtype_t dtype) {
    static char macros_buf[8192];
    int use_native = vk_supports_native_storage(dtype);
    
    if (!use_native && (dtype == ACE_DTYPE_FLOAT16 || dtype == ACE_DTYPE_BFLOAT16)) {
        /* 模拟模式 - 只定义常量，函数动态注入 */
        snprintf(macros_buf, sizeof(macros_buf),
            "/* 模拟模式 - kxxx 函数动态注入 */\n"
            "#define K_ZERO uint16_t(0u)\n"
            "#define K_ONE uint16_t(%s)\n"
            "#define K_NEG_ONE uint16_t(%s)\n",
            (dtype == ACE_DTYPE_FLOAT16) ? "0x3C00u" : "0x3F80u",
            (dtype == ACE_DTYPE_FLOAT16) ? "0xBC00u" : "0xBF80u");
    } else if (!use_native && (dtype == ACE_DTYPE_INT8 || dtype == ACE_DTYPE_UINT8)) {
        snprintf(macros_buf, sizeof(macros_buf),
            "/* INT8/UINT8 模拟 */\n"
            "#define kadd(a, b) uint8_t(((a) + (b)) & 0xFFu)\n"
            "#define ksub(a, b) uint8_t(((a) - (b)) & 0xFFu)\n"
            "#define kmul(a, b) uint8_t(((a) * (b)) & 0xFFu)\n"
            "#define kdiv(a, b) uint8_t(((a) / (b)) & 0xFFu)\n"
            "#define klt(a, b) ((a) < (b))\n"
            "#define kle(a, b) ((a) <= (b))\n"
            "#define kgt(a, b) ((a) > (b))\n"
            "#define kge(a, b) ((a) >= (b))\n"
            "#define keq(a, b) ((a) == (b))\n"
            "#define kne(a, b) ((a) != (b))\n"
            "#define K_ZERO uint8_t(0u)\n"
            "#define K_ONE uint8_t(1u)\n"
            "#define K_NEG_ONE uint8_t(255u)\n");
    } else if (!use_native && dtype == ACE_DTYPE_INT16) {
        snprintf(macros_buf, sizeof(macros_buf),
            "/* INT16 模拟 */\n"
            "#define kadd(a, b) uint16_t(((a) + (b)) & 0xFFFFu)\n"
            "#define ksub(a, b) uint16_t(((a) - (b)) & 0xFFFFu)\n"
            "#define kmul(a, b) uint16_t(((a) * (b)) & 0xFFFFu)\n"
            "#define kdiv(a, b) uint16_t(((a) / (b)) & 0xFFFFu)\n"
            "#define klt(a, b) ((a) < (b))\n"
            "#define kle(a, b) ((a) <= (b))\n"
            "#define kgt(a, b) ((a) > (b))\n"
            "#define kge(a, b) ((a) >= (b))\n"
            "#define keq(a, b) ((a) == (b))\n"
            "#define kne(a, b) ((a) != (b))\n"
            "#define K_ZERO uint16_t(0u)\n"
            "#define K_ONE uint16_t(1u)\n"
            "#define K_NEG_ONE uint16_t(65535u)\n");
    } else {
        /* 原生模式 */
        const char* type_name = vk_get_compute_type_name(dtype);
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

char* vk_translate_to_glsl(const char* name, const char* src, ace_dtype_t dtype,
                           int* n_buffers, int* n_scalars) {
    (void)name;
    const char* body_start = strchr(src, '{');
    const char* body_end = strrchr(src, '}');
    if (!body_start || !body_end) {
        return strdup("#version 450\nlayout(local_size_x=256) in;\nvoid main(){}\n");
    }

    size_t body_len = body_end - body_start - 1;
    int use_native = vk_supports_native_storage(dtype);
    const char* buffer_type = use_native ? vk_get_buffer_type_name(dtype) :
        (dtype == ACE_DTYPE_FLOAT16 || dtype == ACE_DTYPE_BFLOAT16 || dtype == ACE_DTYPE_INT16 ? "uint16_t" :
         dtype == ACE_DTYPE_INT8 || dtype == ACE_DTYPE_UINT8 ? "uint8_t" : "uint");

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

    const char* extensions = vk_get_glsl_extension(dtype);
    const char* converters = get_type_converters(dtype);
    const char* kernel_macros = vk_get_kernel_macros(dtype);

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

    /* 动态检测内核代码中使用的 kxxx 函数 */
    int need_converters = 0;
    if (!use_native && (dtype == ACE_DTYPE_FLOAT16 || dtype == ACE_DTYPE_BFLOAT16)) {
        if (strstr(src, "kadd") || strstr(src, "ksub") || strstr(src, "kmul") ||
            strstr(src, "kdiv") || strstr(src, "klt") || strstr(src, "kle") ||
            strstr(src, "kgt") || strstr(src, "kge") || strstr(src, "keq") ||
            strstr(src, "kne")) {
            need_converters = 1;
        }
    }

    /* 类型定义和内核函数宏 */
    const char* type_defs = "";
    static char type_defs_buf[8192];
    
    if (!use_native && (dtype == ACE_DTYPE_FLOAT16 || dtype == ACE_DTYPE_BFLOAT16)) {
        /* 模拟模式 - 需要启用 16/8 位类型扩展 */
        snprintf(type_defs_buf, sizeof(type_defs_buf),
            "/* 启用 16/8 位类型扩展 */\n"
            "#extension GL_EXT_shader_explicit_arithmetic_types : require\n"
            "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n"
            "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n");
        
        /* 添加类型定义和运算函数 */
        char func_defs[4096] = "";
        
        if (dtype == ACE_DTYPE_BFLOAT16) {
            snprintf(func_defs, sizeof(func_defs),
                "/* BF16 模拟 */\n"
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
        } else if (dtype == ACE_DTYPE_FLOAT16) {
            snprintf(func_defs, sizeof(func_defs),
                "/* FP16 模拟 */\n"
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
        }
        
        /* 合并扩展声明和函数定义 */
        strncat(type_defs_buf, func_defs, sizeof(type_defs_buf) - strlen(type_defs_buf) - 1);
        type_defs = type_defs_buf;
    } else {
        type_defs = kernel_macros;
    }

    char* body = (char*)malloc(body_len + 1);
    strncpy(body, body_start + 1, body_len);
    body[body_len] = '\0';

    /* 动态检测是否需要注入 kxxx 函数 */
    int need_inject = 0;
    if (!use_native && (dtype == ACE_DTYPE_FLOAT16 || dtype == ACE_DTYPE_BFLOAT16)) {
        if (strstr(body, "kadd") || strstr(body, "ksub") || strstr(body, "kmul") ||
            strstr(body, "kdiv") || strstr(body, "klt") || strstr(body, "kle") ||
            strstr(body, "kgt") || strstr(body, "kge")) {
            need_inject = 1;
        }
    }

    size_t defs_len = need_inject ? strlen(type_defs) : 0;
    size_t conv_len = need_converters ? strlen(converters) : 0;
    size_t len = 8192 + strlen(buffers) + strlen(push_constants) + strlen(pc_access) + defs_len + conv_len + strlen(body);
    char* out = (char*)malloc(len);

    snprintf(out, len,
        "#version 450\n"
        "%s"
        "\nlayout(local_size_x = 256) in;\n"
        "%s"
        "%s"
        "%s"
        "%s"
        "%s"
        "#define GID int(gl_GlobalInvocationID.x)\n"
        "#define LID int(gl_LocalInvocationID.x)\n"
        "#define BSIZE 256\n"
        "#define BARRIER() barrier()\n"
        "void main() {\n"
        "%s"
        "}\n",
        extensions,
        need_inject ? type_defs : "",
        buffers,
        push_constants,
        pc_access,
        need_converters ? converters : "",
        body);

    /* 动态注入实际使用到的 kxxx 函数 */
    if (need_inject) {
        inject_used_k_functions(out, len, body, dtype);
    }

    free(body);
    return out;
}
