/**
 * @file cuda_dtype_table.c
 * @brief CUDA 数据类型表实现
 */
#include "cuda_dtype_table.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string.h>

static int g_compute_capability = 0;
static int g_init = 0;

static void init_compute_capability(void) {
    if (g_init) return;
    CUdevice device;
    CUresult result = cuDeviceGet(&device, 0);
    if (result == CUDA_SUCCESS) {
        int major = 0, minor = 0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        g_compute_capability = major * 10 + minor;
    }
    g_init = 1;
}

static int check_native(ace_dtype_t dtype) {
    init_compute_capability();
    switch (dtype) {
        case ACE_DTYPE_FLOAT32: case ACE_DTYPE_FLOAT64:
        case ACE_DTYPE_INT32: case ACE_DTYPE_INT64:
            return 1;
        case ACE_DTYPE_FLOAT16: return (g_compute_capability >= 53);
        case ACE_DTYPE_BFLOAT16: return (g_compute_capability >= 80);
        case ACE_DTYPE_INT8: case ACE_DTYPE_UINT8: return (g_compute_capability >= 61);
        case ACE_DTYPE_INT16: return (g_compute_capability >= 30);
        default: return 1;
    }
}

static const char* FP16_DEF =
    "/* FP16 类型定义和转换函数 */\n"
    "typedef unsigned short half_t;\n"
    "__device__ inline half_t f32_to_f16(float f) {\n"
    "    unsigned int u = __float_as_uint(f);\n"
    "    unsigned int sign = (u >> 16u) & 0x8000u;\n"
    "    unsigned int exp = ((u >> 23u) & 0xFFu) - 112u;\n"
    "    unsigned int frac = (u >> 13u) & 0x3FFu;\n"
    "    if ((u & 0x7FFFFFFFu) == 0u) return (half_t)sign;\n"
    "    if (exp > 30u) return (half_t)(sign | 0x7C00u);\n"
    "    return (half_t)(sign | (exp << 10u) | frac);\n"
    "}\n"
    "__device__ inline float f16_to_f32(half_t h) {\n"
    "    unsigned int sign = (h & 0x8000u) << 16u;\n"
    "    unsigned int exp = (h & 0x7C00u) >> 10u;\n"
    "    unsigned int frac = (h & 0x03FFu) << 13u;\n"
    "    if (exp == 0u) exp = 0;\n"
    "    else if (exp == 31u) exp = 255;\n"
    "    else exp += 127u - 15u;\n"
    "    unsigned int u = sign | (exp << 23u) | frac;\n"
    "    return __uint_as_float(u);\n"
    "}\n";

static const char* BF16_DEF =
    "/* BF16 类型定义和转换函数 */\n"
    "typedef unsigned short bfloat16_t;\n"
    "__device__ inline bfloat16_t f32_to_bf16(float f) {\n"
    "    unsigned int u = __float_as_uint(f);\n"
    "    return (bfloat16_t)((u >> 16u) & 0xFFFFu);\n"
    "}\n"
    "__device__ inline float bf16_to_f32(bfloat16_t h) {\n"
    "    unsigned int u = ((unsigned int)h) << 16u;\n"
    "    return __uint_as_float(u);\n"
    "}\n";

static dtype_info_t g_table[(ACE_DTYPE_BOOL + 1)];

const dtype_info_t* cuda_get_dtype_table(void) {
    if (g_table[0].name) return g_table;

    int native_fp16 = check_native(ACE_DTYPE_FLOAT16);
    int native_bf16 = check_native(ACE_DTYPE_BFLOAT16);
    int native_int8 = check_native(ACE_DTYPE_INT8);
    int native_int16 = check_native(ACE_DTYPE_INT16);

    g_table[ACE_DTYPE_FLOAT32] = (dtype_info_t){
        .dtype = ACE_DTYPE_FLOAT32, .name = "float", .headers = "",
        .type_def = "", .k_zero = "#define K_ZERO float(0)",
        .k_one = "#define K_ONE float(1)", .k_neg_one = "#define K_NEG_ONE float(-1)",
        .size = 4, .needs_emulation = 0
    };
    g_table[ACE_DTYPE_FLOAT64] = (dtype_info_t){
        .dtype = ACE_DTYPE_FLOAT64, .name = "double", .headers = "",
        .type_def = "", .k_zero = "#define K_ZERO double(0)",
        .k_one = "#define K_ONE double(1)", .k_neg_one = "#define K_NEG_ONE double(-1)",
        .size = 8, .needs_emulation = 0
    };
    g_table[ACE_DTYPE_INT32] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT32, .name = "int", .headers = "",
        .type_def = "", .k_zero = "#define K_ZERO int(0)",
        .k_one = "#define K_ONE int(1)", .k_neg_one = "#define K_NEG_ONE int(-1)",
        .size = 4, .needs_emulation = 0
    };
    g_table[ACE_DTYPE_INT64] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT64, .name = "long long", .headers = "",
        .type_def = "", .k_zero = "#define K_ZERO (long long)0",
        .k_one = "#define K_ONE (long long)1", .k_neg_one = "#define K_NEG_ONE (long long)-1",
        .size = 8, .needs_emulation = 0
    };
    g_table[ACE_DTYPE_FLOAT16] = (dtype_info_t){
        .dtype = ACE_DTYPE_FLOAT16, .name = native_fp16 ? "half" : "unsigned short",
        .headers = "#include <cuda_fp16.h>\n",
        .type_def = native_fp16 ? "" : FP16_DEF,
        .k_zero = native_fp16 ? "#define K_ZERO __float2half(0.0f)" : "#define K_ZERO (unsigned short)0",
        .k_one = native_fp16 ? "#define K_ONE __float2half(1.0f)" : "#define K_ONE (unsigned short)1",
        .k_neg_one = native_fp16 ? "#define K_NEG_ONE __float2half(-1.0f)" : "#define K_NEG_ONE (unsigned short)-1",
        .fn_to_f32 = "f16_to_f32", .fn_from_f32 = "f32_to_f16",
        .size = 2, .needs_emulation = !native_fp16
    };
    g_table[ACE_DTYPE_BFLOAT16] = (dtype_info_t){
        .dtype = ACE_DTYPE_BFLOAT16, .name = native_bf16 ? "__nv_bfloat16" : "unsigned short",
        .headers = "#include <cuda_bf16.h>\n",
        .type_def = native_bf16 ? "" : BF16_DEF,
        .k_zero = native_bf16 ? "#define K_ZERO __float2bfloat16(0.0f)" : "#define K_ZERO (unsigned short)0",
        .k_one = native_bf16 ? "#define K_ONE __float2bfloat16(1.0f)" : "#define K_ONE (unsigned short)1",
        .k_neg_one = native_bf16 ? "#define K_NEG_ONE __float2bfloat16(-1.0f)" : "#define K_NEG_ONE (unsigned short)-1",
        .fn_to_f32 = "bf16_to_f32", .fn_from_f32 = "f32_to_bf16",
        .size = 2, .needs_emulation = !native_bf16
    };
    g_table[ACE_DTYPE_INT8] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT8, .name = "signed char", .headers = "",
        .type_def = "", .k_zero = "#define K_ZERO (signed char)0",
        .k_one = "#define K_ONE (signed char)1", .k_neg_one = "#define K_NEG_ONE (signed char)-1",
        .size = 1, .needs_emulation = !native_int8
    };
    g_table[ACE_DTYPE_UINT8] = (dtype_info_t){
        .dtype = ACE_DTYPE_UINT8, .name = "unsigned char", .headers = "",
        .type_def = "", .k_zero = "#define K_ZERO (unsigned char)0",
        .k_one = "#define K_ONE (unsigned char)1", .k_neg_one = "#define K_NEG_ONE (unsigned char)-1",
        .size = 1, .needs_emulation = !native_int8
    };
    g_table[ACE_DTYPE_INT16] = (dtype_info_t){
        .dtype = ACE_DTYPE_INT16, .name = "short", .headers = "",
        .type_def = "", .k_zero = "#define K_ZERO (short)0",
        .k_one = "#define K_ONE (short)1", .k_neg_one = "#define K_NEG_ONE (short)-1",
        .size = 2, .needs_emulation = !native_int16
    };
    g_table[ACE_DTYPE_BOOL] = (dtype_info_t){
        .dtype = ACE_DTYPE_BOOL, .name = "bool", .headers = "",
        .type_def = "", .k_zero = "#define K_ZERO false",
        .k_one = "#define K_ONE true", .k_neg_one = "#define K_NEG_ONE true",
        .size = 1, .needs_emulation = 0
    };

    return g_table;
}
