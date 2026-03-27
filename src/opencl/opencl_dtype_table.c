/**
 * @file opencl_dtype_table.c
 * @brief OpenCL 数据类型表实现
 */
#include "opencl_dtype_table.h"
#include "opencl_backend.h"
#include <string.h>

static dtype_info_t g_table[(ACE_DTYPE_BOOL + 1)];

static const char* FP16_DEF =
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
    "}\n";

static const char* BF16_DEF =
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
    "}\n";

const dtype_info_t* opencl_get_dtype_table(void) {
    if (g_table[0].name) return g_table;

    int native_fp16 = g_device_exts.has_fp16;
    int native_fp64 = g_device_exts.has_fp64;
    int native_int64 = g_device_exts.has_int64;
    int native_int8 = g_device_exts.has_int8 && g_device_exts.has_8bit_storage;
    int native_int16 = g_device_exts.has_int16 && g_device_exts.has_16bit_storage;

    g_table[ACE_DTYPE_FLOAT32] = (dtype_info_t){
        .dtype = ACE_DTYPE_FLOAT32, .name = "float", .extension = "",
        .type_def = "", .k_zero = "#define K_ZERO (float)0",
        .k_one = "#define K_ONE (float)1", .k_neg_one = "#define K_NEG_ONE (float)-1",
        .size = 4, .needs_emulation = 0
    };
    g_table[ACE_DTYPE_FLOAT64] = (dtype_info_t){
        .dtype = ACE_DTYPE_FLOAT64, .name = "double",
        .extension = native_fp64 ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" : "",
        .type_def = "", .k_zero = "#define K_ZERO (double)0",
        .k_one = "#define K_ONE (double)1", .k_neg_one = "#define K_NEG_ONE (double)-1",
        .size = 8, .needs_emulation = !native_fp64
    };
    g_table[ACE_DTYPE_INT32] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT32, .name = "int", .extension = "",
        .type_def = "", .k_zero = "#define K_ZERO (int)0",
        .k_one = "#define K_ONE (int)1", .k_neg_one = "#define K_NEG_ONE (int)-1",
        .size = 4, .needs_emulation = 0
    };
    g_table[ACE_DTYPE_INT64] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT64, .name = "long", .extension = "",
        .type_def = "", .k_zero = "#define K_ZERO (long)0",
        .k_one = "#define K_ONE (long)1", .k_neg_one = "#define K_NEG_ONE (long)-1",
        .size = 8, .needs_emulation = !native_int64
    };
    g_table[ACE_DTYPE_FLOAT16] = (dtype_info_t){
        .dtype = ACE_DTYPE_FLOAT16, .name = native_fp16 ? "half" : "ushort",
        .extension = native_fp16 ? "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n" : "",
        .type_def = native_fp16 ? "" : FP16_DEF,
        .k_zero = "#define K_ZERO (ushort)0",
        .k_one = "#define K_ONE (ushort)1", .k_neg_one = "#define K_NEG_ONE (ushort)-1",
        .fn_to_f32 = "f16_to_f32", .fn_from_f32 = "f32_to_f16",
        .size = 2, .needs_emulation = !native_fp16
    };
    g_table[ACE_DTYPE_BFLOAT16] = (dtype_info_t){
        .dtype = ACE_DTYPE_BFLOAT16, .name = "ushort", .extension = "",
        .type_def = BF16_DEF,
        .k_zero = "#define K_ZERO (ushort)0",
        .k_one = "#define K_ONE (ushort)1", .k_neg_one = "#define K_NEG_ONE (ushort)-1",
        .fn_to_f32 = "bf16_to_f32", .fn_from_f32 = "f32_to_bf16",
        .size = 2, .needs_emulation = 1
    };
    g_table[ACE_DTYPE_INT8] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT8, .name = "char", .extension = "",
        .type_def = "", .k_zero = "#define K_ZERO (char)0",
        .k_one = "#define K_ONE (char)1", .k_neg_one = "#define K_NEG_ONE (char)-1",
        .size = 1, .needs_emulation = !native_int8
    };
    g_table[ACE_DTYPE_UINT8] = (dtype_info_t){
        .dtype = ACE_DTYPE_UINT8, .name = "uchar", .extension = "",
        .type_def = "", .k_zero = "#define K_ZERO (uchar)0",
        .k_one = "#define K_ONE (uchar)1", .k_neg_one = "#define K_NEG_ONE (uchar)-1",
        .size = 1, .needs_emulation = !native_int8
    };
    g_table[ACE_DTYPE_INT16] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT16, .name = "short", .extension = "",
        .type_def = "", .k_zero = "#define K_ZERO (short)0",
        .k_one = "#define K_ONE (short)1", .k_neg_one = "#define K_NEG_ONE (short)-1",
        .size = 2, .needs_emulation = !native_int16
    };
    g_table[ACE_DTYPE_BOOL] = (dtype_info_t){
        .dtype = ACE_DTYPE_BOOL, .name = "bool", .extension = "",
        .type_def = "", .k_zero = "#define K_ZERO false",
        .k_one = "#define K_ONE true", .k_neg_one = "#define K_NEG_ONE true",
        .size = 1, .needs_emulation = 0
    };

    return g_table;
}
