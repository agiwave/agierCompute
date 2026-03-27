/**
 * @file cuda_dtype_table.h
 * @brief CUDA 数据类型表 - 设备级别
 * 
 * 每个设备拥有独立的类型表，根据设备的计算能力动态生成。
 */
#ifndef CUDA_DTYPE_TABLE_H
#define CUDA_DTYPE_TABLE_H

#include "ace.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 数据类型信息
 */
typedef struct {
    ace_dtype_t dtype;
    const char* name;
    const char* headers;
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
 * @brief 设备级别的数据类型表
 */
typedef struct {
    dtype_info_t entries[ACE_DTYPE_BOOL + 1];
    int compute_capability;  /* 设备计算能力 */
} cuda_dtype_table_t;

/**
 * @brief 根据设备计算能力初始化类型表
 * @param table 要初始化的类型表
 * @param compute_major 计算能力主版本
 * @param compute_minor 计算能力次版本
 */
void cuda_dtype_table_init(cuda_dtype_table_t* table, int compute_major, int compute_minor);

/**
 * @brief 获取类型信息
 * @param table 设备的类型表
 * @param dtype 数据类型
 * @return 类型信息指针
 */
static inline const dtype_info_t* cuda_dtype_info(const cuda_dtype_table_t* table, ace_dtype_t dtype) {
    if (!table || dtype < 0 || dtype > ACE_DTYPE_BOOL) return &table->entries[0];
    return &table->entries[dtype];
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_DTYPE_TABLE_H */
