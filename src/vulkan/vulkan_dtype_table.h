/**
 * @file vulkan_dtype_table.h
 * @brief Vulkan 数据类型表 - 设备级别
 * 
 * 每个设备拥有独立的类型表，根据设备特性动态生成。
 */
#ifndef VULKAN_DTYPE_TABLE_H
#define VULKAN_DTYPE_TABLE_H

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
    const char* extensions;
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
 * @brief 设备特性支持状态
 */
typedef struct {
    int has_float16;
    int has_int8;
    int has_int16;
    int has_16bit_storage;
    int has_8bit_storage;
    int has_bfloat16;
    /* 64 位类型支持 */
    int has_float64;      /* shaderFloat64 支持 */
    int has_int64;        /* shaderInt64 支持 */
    int has_64bit_storage; /* storageBuffer64BitAccess 支持 */
} vk_device_features_t;

/**
 * @brief 设备级别的数据类型表
 */
typedef struct {
    dtype_info_t entries[ACE_DTYPE_BOOL + 1];
    vk_device_features_t features;  /* 设备特性状态 */
} vk_dtype_table_t;

/**
 * @brief 根据设备特性初始化类型表
 * @param table 要初始化的类型表
 * @param features 设备特性支持状态
 */
void vk_dtype_table_init(vk_dtype_table_t* table, const vk_device_features_t* features);

/**
 * @brief 获取类型信息
 * @param table 设备的类型表
 * @param dtype 数据类型
 * @return 类型信息指针
 */
static inline const dtype_info_t* vk_dtype_info(const vk_dtype_table_t* table, ace_dtype_t dtype) {
    if (!table || dtype < 0 || dtype > ACE_DTYPE_BOOL) return &table->entries[0];
    return &table->entries[dtype];
}

#ifdef __cplusplus
}
#endif

#endif /* VULKAN_DTYPE_TABLE_H */