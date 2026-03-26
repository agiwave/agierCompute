/**
 * @file vulkan_type_utils.c
 * @brief Vulkan type translation utilities
 */
#include "vulkan_backend.h"

#ifdef VULKAN_AVAILABLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 全局设备特性 */
static vk_device_features_t g_device_features = {0};

void vk_detect_device_features(VkPhysicalDevice physical_device) {
    /* 检测 Vulkan 1.0 基础特性 */
    VkPhysicalDeviceFeatures features10 = {0};
    vkGetPhysicalDeviceFeatures(physical_device, &features10);
    
    /* 检测 Vulkan 1.2 特性 */
    VkPhysicalDeviceVulkan12Features features12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
    };

    VkPhysicalDeviceFeatures2 features2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &features12
    };

    vkGetPhysicalDeviceFeatures2(physical_device, &features2);

    g_device_features.has_float16 = features12.shaderFloat16;
    g_device_features.has_int8 = features12.shaderInt8;
    g_device_features.has_int16 = features10.shaderInt16;  /* shaderInt16 在 Vulkan 1.0 特性中 */

    /* 检测 16-bit storage 扩展 */
    VkPhysicalDevice16BitStorageFeatures storage16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES
    };
    features2.pNext = &storage16;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);

    g_device_features.has_16bit_storage = storage16.storageBuffer16BitAccess;

    /* 检测 8-bit storage 扩展 */
    VkPhysicalDevice8BitStorageFeatures storage8 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES
    };
    features2.pNext = &storage8;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);

    g_device_features.has_8bit_storage = storage8.storageBuffer8BitAccess;

    /* 检测 bfloat16 扩展 - 使用正确的结构体名称 */
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR
    };
    features2.pNext = &bf16_features;
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);

    g_device_features.has_bfloat16 = bf16_features.shaderBFloat16Type;

    printf("[Vulkan] Device features: FP16=%d, INT8=%d, INT16=%d, 16bit_storage=%d, 8bit_storage=%d, BF16=%d\n",
           g_device_features.has_float16,
           g_device_features.has_int8,
           g_device_features.has_int16,
           g_device_features.has_16bit_storage,
           g_device_features.has_8bit_storage,
           g_device_features.has_bfloat16);
    
    /* 打印详细支持信息 */
    printf("[Vulkan] Native type support:\n");
    printf("  - FLOAT16: %s\n", (g_device_features.has_float16 && g_device_features.has_16bit_storage) ? "YES" : "NO (using emulation)");
    printf("  - BFLOAT16: %s\n", (g_device_features.has_bfloat16 && g_device_features.has_16bit_storage) ? "YES" : "NO (using emulation)");
    printf("  - INT8: %s\n", (g_device_features.has_int8 && g_device_features.has_8bit_storage) ? "YES" : "NO (using emulation)");
    printf("  - INT16: %s\n", (g_device_features.has_int16 && g_device_features.has_16bit_storage) ? "YES" : "NO (using emulation)");
}

int vk_supports_native_storage(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT16:
            return g_device_features.has_float16 && g_device_features.has_16bit_storage;
        case ACE_DTYPE_BFLOAT16:
            return g_device_features.has_bfloat16 && g_device_features.has_16bit_storage;
        case ACE_DTYPE_INT8:
        case ACE_DTYPE_UINT8:
            return g_device_features.has_int8 && g_device_features.has_8bit_storage;
        case ACE_DTYPE_INT16:
            return g_device_features.has_int16 && g_device_features.has_16bit_storage;
        case ACE_DTYPE_INT64:
            return 1;
        default:
            return 1;
    }
}

const char* vk_get_buffer_type_name(ace_dtype_t dtype) {
    if (vk_supports_native_storage(dtype)) {
        switch (dtype) {
            case ACE_DTYPE_FLOAT32:  return "float";
            case ACE_DTYPE_FLOAT64:  return "double";
            case ACE_DTYPE_INT32:    return "int";
            case ACE_DTYPE_INT64:    return "int64_t";
            case ACE_DTYPE_INT8:     return "int8_t";
            case ACE_DTYPE_UINT8:    return "uint8_t";
            case ACE_DTYPE_INT16:    return "int16_t";
            case ACE_DTYPE_FLOAT16:  return "float16_t";
            case ACE_DTYPE_BFLOAT16: return "bfloat16_t";
            default:                 return "float";
        }
    } else {
        switch (dtype) {
            case ACE_DTYPE_FLOAT16:
            case ACE_DTYPE_BFLOAT16:
            case ACE_DTYPE_INT16:
            case ACE_DTYPE_UINT8:
            case ACE_DTYPE_INT8:
                return "uint";
            case ACE_DTYPE_INT64:
                return "uint";
            default:
                return "float";
        }
    }
}

const char* vk_get_compute_type_name(ace_dtype_t dtype) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32:  return "float";
        case ACE_DTYPE_FLOAT64:  return "double";
        case ACE_DTYPE_INT32:    return "int";
        case ACE_DTYPE_INT64:    return "int64_t";
        case ACE_DTYPE_INT8:     return "int8_t";
        case ACE_DTYPE_UINT8:    return "uint8_t";
        case ACE_DTYPE_INT16:    return "int16_t";
        case ACE_DTYPE_FLOAT16:  return "float16_t";
        case ACE_DTYPE_BFLOAT16: return "bfloat16_t";
        default:                 return "float";
    }
}

int vk_is_float_dtype(ace_dtype_t dtype) {
    return (dtype == ACE_DTYPE_FLOAT32 || dtype == ACE_DTYPE_FLOAT64 ||
            dtype == ACE_DTYPE_FLOAT16 || dtype == ACE_DTYPE_BFLOAT16);
}

const char* vk_get_glsl_extension(ace_dtype_t dtype) {
    /* 检查是否支持原生类型，不支持则返回空字符串（使用 uint 模拟） */
    if (!vk_supports_native_storage(dtype)) {
        return "";  /* 使用 uint 模拟，不需要扩展 */
    }
    
    /* 设备支持原生类型，返回相应的扩展声明 */
    switch (dtype) {
        case ACE_DTYPE_FLOAT64:
            return "#extension GL_ARB_gpu_shader_fp64 : require\n";
        case ACE_DTYPE_INT64:
            return "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
        case ACE_DTYPE_FLOAT16:
            return "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
        case ACE_DTYPE_BFLOAT16:
            return "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
        case ACE_DTYPE_INT8:
        case ACE_DTYPE_UINT8:
            return "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
        case ACE_DTYPE_INT16:
            return "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
        default:
            return "";
    }
}

char* vk_translate_to_glsl(const char* name, const char* src, ace_dtype_t dtype, int* n_buffers, int* n_scalars) {
    (void)name;  /* 可能未使用 */
    const char* body_start = strchr(src, '{');
    const char* body_end = strrchr(src, '}');
    if (!body_start || !body_end) {
        return strdup("#version 450\nlayout(local_size_x=256) in;\nvoid main(){}\n");
    }

    size_t body_len = body_end - body_start - 1;
    const char* buffer_type = vk_get_buffer_type_name(dtype);
    int use_native = vk_supports_native_storage(dtype);

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
            while (*p == ' ' || *p == '\t' || *p == '\n') p++;
            if (*p == ')') break;

            const char* star = strchr(p, '*');
            const char* comma = strchr(p, ',');
            const char* paren = strchr(p, ')');
            const char* end = comma ? (paren && paren < comma ? paren : comma) : paren;

            if (star && (!end || star < end)) {
                params[n_params].is_buffer = 1;
                const char* name_start = star + 1;
                while (*name_start == ' ' || *name_start == '\t') name_start++;
                const char* name_end = end;
                while (name_end > name_start && (*(name_end-1) == ' ' || *(name_end-1) == '\t')) name_end--;
                size_t len = name_end - name_start;
                if (len >= sizeof(params[n_params].name)) len = sizeof(params[n_params].name) - 1;
                strncpy(params[n_params].name, name_start, len);
                params[n_params].name[len] = '\0';
                (*n_buffers)++;
            } else {
                params[n_params].is_buffer = 0;
                const char* type_end = end;
                while (type_end > p && *(type_end-1) != ' ' && *(type_end-1) != '\t') type_end--;
                if (type_end == p) type_end = end;
                const char* name_start = type_end;
                while (*name_start == ' ' || *name_start == '\t') name_start++;
                const char* name_end = end;
                while (name_end > name_start && (*(name_end-1) == ' ' || *(name_end-1) == '\t')) name_end--;
                size_t len = name_end - name_start;
                if (len >= sizeof(params[n_params].name)) len = sizeof(params[n_params].name) - 1;
                strncpy(params[n_params].name, name_start, len);
                params[n_params].name[len] = '\0';
                (*n_scalars)++;
            }
            n_params++;

            if (comma && comma < paren) p = comma + 1;
            else p = end;
        }
    }

    if (*n_buffers == 0) *n_buffers = 1;
    if (*n_buffers > 8) *n_buffers = 8;

    char push_constants[1024] = "";
    char pc_access[1024] = "";

    if (*n_scalars > 0) {
        strcpy(push_constants, "layout(push_constant) uniform PC {\n");
        int scalar_idx = 0;
        for (int i = 0; i < n_params; i++) {
            if (!params[i].is_buffer) {
                char line[128];
                /* 第一个标量参数 (n) 总是 int，后续使用 uint（模拟模式）或原生类型 */
                const char* scalar_type = (scalar_idx == 0) ? "int" : (use_native ? buffer_type : "uint");
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
            /* 模拟模式下使用 uint，原生模式下使用原生类型 */
            const char* actual_type = use_native ? buffer_type : "uint";
            snprintf(buf_decl, sizeof(buf_decl),
                "layout(binding = %d, std430) buffer B%d { %s d%d[]; };\n",
                buf_idx, buf_idx, actual_type, buf_idx);
            strcat(buffers, buf_decl);
            char def[128];
            snprintf(def, sizeof(def), "#define %s d%d\n", params[i].name, buf_idx);
            strcat(buffers, def);
            buf_idx++;
        }
    }

    char* body = (char*)malloc(body_len + 1);
    strncpy(body, body_start + 1, body_len);
    body[body_len] = '\0';

    /* 模拟模式下，替换内核代码中的运算为函数调用 */
    if (!use_native) {
        const char* add_func = NULL;
        const char* mul_func = NULL;
        const char* sub_func = NULL;
        const char* div_func = NULL;
        
        if (dtype == ACE_DTYPE_BFLOAT16) {
            add_func = "bf16_add"; mul_func = "bf16_mul"; sub_func = "bf16_sub"; div_func = "bf16_div";
        } else if (dtype == ACE_DTYPE_FLOAT16) {
            add_func = "f16_add"; mul_func = "f16_mul"; sub_func = "f16_sub"; div_func = "f16_div";
        } else if (dtype == ACE_DTYPE_INT8 || dtype == ACE_DTYPE_UINT8) {
            add_func = "int8_add"; mul_func = "int8_mul"; sub_func = "int8_sub";
        } else if (dtype == ACE_DTYPE_INT16) {
            add_func = "int16_add"; mul_func = "int16_mul"; sub_func = "int16_sub";
        }
        
        /* 简单替换：将 a + b 替换为 func(a, b) - 只适用于简单表达式 */
        /* 这是一个简化实现，复杂表达式需要更复杂的解析 */
        if (add_func) {
            char* p = body;
            while ((p = strstr(p, " + ")) != NULL) {
                /* 查找操作数 */
                char* op1_end = p - 1;
                while (op1_end > body && (*op1_end == ' ' || *op1_end == '\t')) op1_end--;
                char* op1_start = op1_end;
                while (op1_start > body && *op1_start != ' ' && *op1_start != '\t' && 
                       *op1_start != '(' && *op1_start != ',' && *op1_start != ';') op1_start--;
                if (*op1_start == '(' || *op1_start == ',' || *op1_start == ';') op1_start++;
                
                char* op2_start = p + 3;
                while (*op2_start == ' ' || *op2_start == '\t') op2_start++;
                char* op2_end = op2_start;
                while (*op2_end && *op2_end != ' ' && *op2_end != '\t' && *op2_end != ';' && 
                       *op2_end != ')' && *op2_end != ',' && *op2_end != '[') op2_end++;
                
                size_t op1_len = op1_end - op1_start + 1;
                size_t op2_len = op2_end - op2_start;
                size_t func_len = strlen(add_func);
                
                /* 构建新字符串 */
                char* new_body = malloc(strlen(body) + func_len + 10);
                strncpy(new_body, body, op1_start - body);
                sprintf(new_body + (op1_start - body), "%s(", add_func);
                strncpy(new_body + (op1_start - body) + func_len + 1, op1_start, op1_len);
                strcpy(new_body + (op1_start - body) + func_len + 1 + op1_len, ", ");
                strncpy(new_body + strlen(new_body), op2_start, op2_len);
                strcpy(new_body + strlen(new_body), ")");
                strcpy(new_body + strlen(new_body), op2_end);
                
                free(body);
                body = new_body;
                p = body + (op1_start - body) + func_len + 1 + op1_len + 2 + op2_len + 1;
            }
        }
        if (mul_func) {
            char* p = body;
            while ((p = strstr(p, " * ")) != NULL) {
                char* op1_end = p - 1;
                while (op1_end > body && (*op1_end == ' ' || *op1_end == '\t')) op1_end--;
                char* op1_start = op1_end;
                while (op1_start > body && *op1_start != ' ' && *op1_start != '\t' && 
                       *op1_start != '(' && *op1_start != ',' && *op1_start != ';') op1_start--;
                if (*op1_start == '(' || *op1_start == ',' || *op1_start == ';') op1_start++;
                
                char* op2_start = p + 3;
                while (*op2_start == ' ' || *op2_start == '\t') op2_start++;
                char* op2_end = op2_start;
                while (*op2_end && *op2_end != ' ' && *op2_end != '\t' && *op2_end != ';' && 
                       *op2_end != ')' && *op2_end != ',' && *op2_end != '[') op2_end++;
                
                size_t op1_len = op1_end - op1_start + 1;
                size_t op2_len = op2_end - op2_start;
                size_t func_len = strlen(mul_func);
                
                char* new_body = malloc(strlen(body) + func_len + 10);
                strncpy(new_body, body, op1_start - body);
                sprintf(new_body + (op1_start - body), "%s(", mul_func);
                strncpy(new_body + (op1_start - body) + func_len + 1, op1_start, op1_len);
                strcpy(new_body + (op1_start - body) + func_len + 1 + op1_len, ", ");
                strncpy(new_body + strlen(new_body), op2_start, op2_len);
                strcpy(new_body + strlen(new_body), ")");
                strcpy(new_body + strlen(new_body), op2_end);
                
                free(body);
                body = new_body;
                p = body + (op1_start - body) + func_len + 1 + op1_len + 2 + op2_len + 1;
            }
        }
    }

    const char* extensions = vk_get_glsl_extension(dtype);
    /* use_native 已在前面定义 */

    /* BF16/FP16/INT8/INT16 类型定义和辅助函数 - 当使用模拟时需要 */
    const char* type_defs = "";
    static char type_defs_buf[4096];
    
    if (!use_native) {
        /* 使用模拟实现 - 定义自定义类型和辅助函数 */
        if (dtype == ACE_DTYPE_BFLOAT16) {
            snprintf(type_defs_buf, sizeof(type_defs_buf),
                "/* BF16 模拟 - 使用 uint 存储，提供转换和运算函数 */\n"
                "float bf16_to_f32(uint x) {\n"
                "    int sign = int((x >> 15) & 0x1u);\n"
                "    int exp = int((x >> 7) & 0xFFu);\n"
                "    int man = int(x & 0x7Fu);\n"
                "    if (exp == 0) return sign != 0 ? -0.0 : 0.0;\n"
                "    if (exp == 255) return sign != 0 ? -1.0/0.0 : 1.0/0.0;\n"
                "    int result = (sign << 31) | ((exp + 112) << 23) | (man << 16);\n"
                "    return uintBitsToFloat(uint(result));\n"
                "}\n"
                "uint f32_to_bf16(float x) {\n"
                "    uint u = floatBitsToUint(x);\n"
                "    uint sign = (u >> 16) & 0x8000u;\n"
                "    uint exp = ((u >> 23) & 0xFFu) - 112u;\n"
                "    uint man = (u >> 16) & 0x7Fu;\n"
                "    if ((u & 0x7FFFFFFFu) == 0u) return sign;\n"
                "    if (exp > 254u) return sign | 0x7F80u;\n"
                "    return sign | (exp << 7u) | man;\n"
                "}\n"
                "/* BF16 运算 - 转换为 float 计算后转回 */\n"
                "uint bf16_add(uint a, uint b) { return f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b)); }\n"
                "uint bf16_mul(uint a, uint b) { return f32_to_bf16(bf16_to_f32(a) * bf16_to_f32(b)); }\n"
                "uint bf16_sub(uint a, uint b) { return f32_to_bf16(bf16_to_f32(a) - bf16_to_f32(b)); }\n"
                "uint bf16_div(uint a, uint b) { return f32_to_bf16(bf16_to_f32(a) / bf16_to_f32(b)); }\n"
                "/* 类型别名 - 使用 uint 作为底层存储 */\n"
                "typedef uint bfloat16_t;\n");
        } else if (dtype == ACE_DTYPE_FLOAT16) {
            snprintf(type_defs_buf, sizeof(type_defs_buf),
                "/* FP16 模拟 - 使用 uint 存储，提供转换和运算函数 */\n"
                "float f16_to_f32(uint x) {\n"
                "    int sign = int((x >> 15) & 0x1u);\n"
                "    int exp = int((x >> 10) & 0x1Fu);\n"
                "    int man = int(x & 0x3FFu);\n"
                "    if (exp == 0) return sign != 0 ? -0.0 : 0.0;\n"
                "    if (exp == 31) return sign != 0 ? -1.0/0.0 : 1.0/0.0;\n"
                "    int result = (sign << 31) | ((exp + 112) << 23) | (man << 13);\n"
                "    return uintBitsToFloat(uint(result));\n"
                "}\n"
                "uint f32_to_f16(float x) {\n"
                "    uint u = floatBitsToUint(x);\n"
                "    uint sign = (u >> 16) & 0x8000u;\n"
                "    uint exp = ((u >> 23) & 0xFFu) - 112u;\n"
                "    uint man = (u >> 13) & 0x3FFu;\n"
                "    if ((u & 0x7FFFFFFFu) == 0u) return sign;\n"
                "    if (exp > 30u) return sign | 0x7C00u;\n"
                "    return sign | (exp << 10u) | man;\n"
                "}\n"
                "/* FP16 运算 */\n"
                "uint f16_add(uint a, uint b) { return f32_to_f16(f16_to_f32(a) + f16_to_f32(b)); }\n"
                "uint f16_mul(uint a, uint b) { return f32_to_f16(f16_to_f32(a) * f16_to_f32(b)); }\n"
                "uint f16_sub(uint a, uint b) { return f32_to_f16(f16_to_f32(a) - f16_to_f32(b)); }\n"
                "uint f16_div(uint a, uint b) { return f32_to_f16(f16_to_f32(a) / f16_to_f32(b)); }\n"
                "/* 类型别名 */\n"
                "typedef uint half;\n");
        } else if (dtype == ACE_DTYPE_INT8 || dtype == ACE_DTYPE_UINT8) {
            snprintf(type_defs_buf, sizeof(type_defs_buf),
                "/* INT8/UINT8 模拟 - 使用 uint 存储，低 8 位有效 */\n"
                "uint int8_add(uint a, uint b) { return (a + b) & 0xFFu; }\n"
                "uint int8_mul(uint a, uint b) { return (a * b) & 0xFFu; }\n"
                "uint int8_sub(uint a, uint b) { return (a - b) & 0xFFu; }\n"
                "typedef uint int8_t;\n");
        } else if (dtype == ACE_DTYPE_INT16) {
            snprintf(type_defs_buf, sizeof(type_defs_buf),
                "/* INT16 模拟 - 使用 uint 存储，低 16 位有效 */\n"
                "uint int16_add(uint a, uint b) { return (a + b) & 0xFFFFu; }\n"
                "uint int16_mul(uint a, uint b) { return (a * b) & 0xFFFFu; }\n"
                "uint int16_sub(uint a, uint b) { return (a - b) & 0xFFFFu; }\n"
                "typedef uint int16_t;\n");
        }
        type_defs = type_defs_buf;
    } else if (dtype == ACE_DTYPE_BFLOAT16) {
        /* 原生 BF16 支持 */
        snprintf(type_defs_buf, sizeof(type_defs_buf),
            "/* 原生 BF16 辅助函数 */\n"
            "float bf16_to_f32(bfloat16_t x) {\n"
            "    int sign = int((uint(x) >> 15) & 0x1u);\n"
            "    int exp = int((uint(x) >> 7) & 0xFFu);\n"
            "    int man = int(uint(x) & 0x7Fu);\n"
            "    if (exp == 0) return sign != 0 ? -0.0 : 0.0;\n"
            "    if (exp == 255) return sign != 0 ? -1.0/0.0 : 1.0/0.0;\n"
            "    int result = (sign << 31) | ((exp + 112) << 23) | (man << 16);\n"
            "    return uintBitsToFloat(uint(result));\n"
            "}\n");
        type_defs = type_defs_buf;
    }

    size_t len = 8192 + strlen(buffers) + strlen(push_constants) + strlen(pc_access) + strlen(type_defs) + strlen(body);
    char* out = (char*)malloc(len);

    snprintf(out, len,
        "#version 450\n"
        "%s"
        "%s"
        "%s"
        "%s"
        "layout(local_size_x = 256) in;\n"
        "%s"
        "#define GID int(gl_GlobalInvocationID.x)\n"
        "#define LID int(gl_LocalInvocationID.x)\n"
        "#define BSIZE 256\n"
        "#define BARRIER() barrier()\n"
        "void main() {\n"
        "%s"
        "}\n",
        extensions,
        type_defs,
        buffers,
        push_constants,
        pc_access,
        body);

    free(body);
    return out;
}

#endif /* VULKAN_AVAILABLE */
