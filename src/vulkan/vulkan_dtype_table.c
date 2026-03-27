/**
 * @file vulkan_dtype_table.c
 * @brief Vulkan 数据类型表实现 - 设备级别
 */
#include "vulkan_dtype_table.h"
#include <string.h>

/* 类型定义和转换函数 */
static const char* FP16_DEF =
    "float f16_to_f32(uint16_t h) {\n"
    "    uint sign = (h & 0x8000u) << 16u;\n"
    "    uint e = (h & 0x7C00u) >> 10u;\n"
    "    uint f = (h & 0x03FFu) << 13u;\n"
    "    if (e == 0u) return sign != 0u ? -0.0 : 0.0;\n"
    "    if (e == 31u) return sign != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u);\n"
    "    e += 112u;\n"
    "    return uintBitsToFloat(sign | (e << 23u) | f);\n"
    "}\n"
    "uint16_t f32_to_f16(float x) {\n"
    "    uint u = floatBitsToUint(x);\n"
    "    uint sign = (u >> 16u) & 0x8000u;\n"
    "    uint e = ((u >> 23u) & 0xFFu) - 112u;\n"
    "    uint mf = (u >> 13u) & 0x3FFu;\n"
    "    if ((u & 0x7FFFFFFFu) == 0u) return uint16_t(sign);\n"
    "    if (e > 30u) return uint16_t(sign | 0x7C00u);\n"
    "    return uint16_t(sign | (e << 10u) | mf);\n"
    "}\n";

static const char* BF16_DEF =
    "float bf16_to_f32(uint16_t x) {\n"
    "    uint sign = (x >> 15) & 0x1u;\n"
    "    uint e = (x >> 7) & 0xFFu;\n"
    "    uint mf = x & 0x7Fu;\n"
    "    if (e == 0u) return sign != 0u ? -0.0 : 0.0;\n"
    "    if (e == 255u) return sign != 0u ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u);\n"
    "    uint result = (sign << 31u) | (e << 23u) | (mf << 16u);\n"
    "    return uintBitsToFloat(result);\n"
    "}\n"
    "uint16_t f32_to_bf16(float x) {\n"
    "    uint u = floatBitsToUint(x);\n"
    "    uint sign = (u >> 16u) & 0x8000u;\n"
    "    uint e = (u >> 23u) & 0xFFu;\n"
    "    uint mf = (u >> 16u) & 0x7Fu;\n"
    "    if ((u & 0x7FFFFFFFu) == 0u) return uint16_t(sign);\n"
    "    if (e == 255u) return uint16_t(sign | 0x7F80u);\n"
    "    return uint16_t(sign | (e << 7u) | mf);\n"
    "}\n";

static const char* EXT_NONE = "";
static const char* EXT_FP16 = "#extension GL_EXT_shader_16bit_storage : require\n"
                              "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
static const char* EXT_FP64 = "#extension GL_EXT_shader_explicit_arithmetic_types : require\n"
                              "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
static const char* EXT_INT64 = "#extension GL_EXT_shader_explicit_arithmetic_types : require\n"
                               "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
static const char* EXT_INT8 = "#extension GL_EXT_shader_8bit_storage : require\n"
                              "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
static const char* EXT_INT16 = "#extension GL_EXT_shader_16bit_storage : require\n"
                               "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
static const char* EXT_EMU = "#extension GL_EXT_shader_explicit_arithmetic_types : require\n"
                             "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n"
                             "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";

static int check_native(ace_dtype_t dtype, const vk_device_features_t* features) {
    switch (dtype) {
        case ACE_DTYPE_FLOAT32: case ACE_DTYPE_INT32: return 1;
        case ACE_DTYPE_FLOAT64: return features->has_float64 && features->has_64bit_storage;
        case ACE_DTYPE_INT64: return features->has_int64 && features->has_64bit_storage;
        case ACE_DTYPE_FLOAT16: return features->has_float16 && features->has_16bit_storage;
        case ACE_DTYPE_BFLOAT16: return 0;
        case ACE_DTYPE_INT8: case ACE_DTYPE_UINT8: return features->has_int8 && features->has_8bit_storage;
        case ACE_DTYPE_INT16: return features->has_int16 && features->has_16bit_storage;
        default: return 1;
    }
}

void vk_dtype_table_init(vk_dtype_table_t* table, const vk_device_features_t* features) {
    if (!table || !features) return;
    
    table->features = *features;

    int native_fp64 = check_native(ACE_DTYPE_FLOAT64, features);
    int native_int64 = check_native(ACE_DTYPE_INT64, features);
    int native_fp16 = check_native(ACE_DTYPE_FLOAT16, features);
    int native_int8 = check_native(ACE_DTYPE_INT8, features);
    int native_int16 = check_native(ACE_DTYPE_INT16, features);

    table->entries[ACE_DTYPE_FLOAT32] = (dtype_info_t){
        .dtype = ACE_DTYPE_FLOAT32, .name = "float", .extensions = EXT_NONE,
        .type_def = EXT_NONE, .k_zero = "#define K_ZERO float(0)",
        .k_one = "#define K_ONE float(1)", .k_neg_one = "#define K_NEG_ONE float(-1)",
        .size = 4, .needs_emulation = 0
    };
    table->entries[ACE_DTYPE_FLOAT64] = (dtype_info_t){
        .dtype = ACE_DTYPE_FLOAT64, .name = "double", .extensions = EXT_FP64,
        .type_def = EXT_NONE, .k_zero = "#define K_ZERO double(0)",
        .k_one = "#define K_ONE double(1)", .k_neg_one = "#define K_NEG_ONE double(-1)",
        .size = 8, .needs_emulation = !native_fp64
    };
    table->entries[ACE_DTYPE_INT32] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT32, .name = "int", .extensions = EXT_NONE,
        .type_def = EXT_NONE, .k_zero = "#define K_ZERO int(0)",
        .k_one = "#define K_ONE int(1)", .k_neg_one = "#define K_NEG_ONE int(-1)",
        .size = 4, .needs_emulation = 0
    };
    table->entries[ACE_DTYPE_INT64] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT64, .name = "int64_t", .extensions = EXT_INT64,
        .type_def = EXT_NONE, .k_zero = "#define K_ZERO int64_t(0)",
        .k_one = "#define K_ONE int64_t(1)", .k_neg_one = "#define K_NEG_ONE int64_t(-1)",
        .size = 8, .needs_emulation = !native_int64
    };
    table->entries[ACE_DTYPE_FLOAT16] = (dtype_info_t){
        .dtype = ACE_DTYPE_FLOAT16, .name = native_fp16 ? "float16_t" : "uint16_t",
        .extensions = native_fp16 ? EXT_FP16 : EXT_EMU,
        .type_def = FP16_DEF,
        .k_zero = native_fp16 ? "#define K_ZERO float16_t(0)" : "#define K_ZERO uint16_t(0)",
        .k_one = native_fp16 ? "#define K_ONE float16_t(1)" : "#define K_ONE uint16_t(1)",
        .k_neg_one = native_fp16 ? "#define K_NEG_ONE float16_t(-1)" : "#define K_NEG_ONE uint16_t(-1)",
        .fn_to_f32 = "f16_to_f32", .fn_from_f32 = "f32_to_f16",
        .size = 2, .needs_emulation = !native_fp16
    };
    table->entries[ACE_DTYPE_BFLOAT16] = (dtype_info_t){
        .dtype = ACE_DTYPE_BFLOAT16, .name = "uint16_t", .extensions = EXT_EMU,
        .type_def = BF16_DEF, .k_zero = "#define K_ZERO uint16_t(0)",
        .k_one = "#define K_ONE uint16_t(1)", .k_neg_one = "#define K_NEG_ONE uint16_t(-1)",
        .fn_to_f32 = "bf16_to_f32", .fn_from_f32 = "f32_to_bf16",
        .size = 2, .needs_emulation = 1
    };
    table->entries[ACE_DTYPE_INT8] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT8, .name = "int8_t", .extensions = EXT_INT8,
        .type_def = EXT_NONE, .k_zero = "#define K_ZERO int8_t(0)",
        .k_one = "#define K_ONE int8_t(1)", .k_neg_one = "#define K_NEG_ONE int8_t(-1)",
        .size = 1, .needs_emulation = !native_int8
    };
    table->entries[ACE_DTYPE_UINT8] = (dtype_info_t){
        .dtype = ACE_DTYPE_UINT8, .name = "uint8_t", .extensions = EXT_INT8,
        .type_def = EXT_NONE, .k_zero = "#define K_ZERO uint8_t(0)",
        .k_one = "#define K_ONE uint8_t(1)", .k_neg_one = "#define K_NEG_ONE uint8_t(-1)",
        .size = 1, .needs_emulation = !native_int8
    };
    table->entries[ACE_DTYPE_INT16] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT16, .name = "int16_t", .extensions = EXT_INT16,
        .type_def = EXT_NONE, .k_zero = "#define K_ZERO int16_t(0)",
        .k_one = "#define K_ONE int16_t(1)", .k_neg_one = "#define K_NEG_ONE int16_t(-1)",
        .size = 2, .needs_emulation = !native_int16
    };
    table->entries[ACE_DTYPE_BOOL] = (dtype_info_t){
        .dtype = ACE_DTYPE_BOOL, .name = "bool", .extensions = EXT_NONE,
        .type_def = EXT_NONE, .k_zero = "#define K_ZERO false",
        .k_one = "#define K_ONE true", .k_neg_one = "#define K_NEG_ONE true",
        .size = 1, .needs_emulation = 0
    };
}