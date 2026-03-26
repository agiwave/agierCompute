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
    /* shaderInt16 在 Vulkan 1.2 中不可用，需要单独检测 */
    g_device_features.has_int16 = 0;  /* 暂时设为 0 */

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
    const char* body_start = strchr(src, '{');
    const char* body_end = strrchr(src, '}');
    if (!body_start || !body_end) {
        return strdup("#version 450\nlayout(local_size_x=256) in;\nvoid main(){}\n");
    }

    size_t body_len = body_end - body_start - 1;
    const char* buffer_type = vk_get_buffer_type_name(dtype);
    int use_native = vk_supports_native_storage(dtype);
    (void)use_native; /* 暂时未使用 */
    (void)name;

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

    char* body = (char*)malloc(body_len + 1);
    strncpy(body, body_start + 1, body_len);
    body[body_len] = '\0';

    const char* extensions = vk_get_glsl_extension(dtype);

    /* BF16 类型定义和辅助函数 */
    const char* type_defs = "";
    static char bf16_defs[1024];
    if (dtype == ACE_DTYPE_BFLOAT16) {
        snprintf(bf16_defs, sizeof(bf16_defs),
            "/* BF16 辅助函数 - 使用 uint 避免 int16_t 参数问题 */\n"
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
            "}\n");
        type_defs = bf16_defs;
    } else if (dtype == ACE_DTYPE_FLOAT16) {
        type_defs = "";
    }

    size_t len = 8192 + strlen(buffers) + strlen(push_constants) + strlen(pc_access) + strlen(type_defs) + strlen(body);
    char* out = (char*)malloc(len);

    snprintf(out, len,
        "#version 450\n"
        "%s"
        "layout(local_size_x = 256) in;\n"
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
        type_defs,
        buffers,
        push_constants,
        pc_access,
        body);

    free(body);
    return out;
}

#endif /* VULKAN_AVAILABLE */
