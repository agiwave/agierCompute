/**
 * @file opencl_dtype_table.h
 * @brief OpenCL 数据类型表 - 设备级别
 * 
 * 每个设备拥有独立的类型表，根据设备的扩展支持动态生成。
 */
#ifndef OPENCL_DTYPE_TABLE_H
#define OPENCL_DTYPE_TABLE_H

#include "ace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ace_dtype_t dtype;
    const char* name;
    const char* extension;
    const char* type_def;
    const char* k_zero;
    const char* k_one;
    const char* k_neg_one;
    const char* fn_to_f32;
    const char* fn_from_f32;
    size_t size;
    int needs_emulation;
} dtype_info_t;

/**
 * @brief 设备扩展支持状态
 */
typedef struct {
    int has_fp16;              /* cl_khr_fp16 */
    int has_fp64;              /* cl_khr_fp64 */
    int has_int64;             /* native support */
    int has_int8;              /* cl_khr_int8 */
    int has_int16;             /* cl_khr_int16 */
    int has_8bit_storage;      /* cl_khr_8bit_storage */
    int has_16bit_storage;     /* cl_khr_16bit_storage */
} ocl_device_extensions_t;

/**
 * @brief 设备级别的数据类型表
 */
typedef struct {
    dtype_info_t entries[ACE_DTYPE_BOOL + 1];
    ocl_device_extensions_t exts;  /* 设备扩展状态 */
} ocl_dtype_table_t;

/**
 * @brief 根据设备扩展初始化类型表
 * @param table 要初始化的类型表
 * @param exts 设备扩展支持状态
 */
void ocl_dtype_table_init(ocl_dtype_table_t* table, const ocl_device_extensions_t* exts);

/**
 * @brief 获取类型信息
 * @param table 设备的类型表
 * @param dtype 数据类型
 * @return 类型信息指针
 */
static inline const dtype_info_t* ocl_dtype_info(const ocl_dtype_table_t* table, ace_dtype_t dtype) {
    if (!table || dtype < 0 || dtype > ACE_DTYPE_BOOL) return &table->entries[0];
    return &table->entries[dtype];
}

#ifdef __cplusplus
}
#endif

#endif /* OPENCL_DTYPE_TABLE_H */