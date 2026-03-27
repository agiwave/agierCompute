/**
 * @file opencl_dtype_table.h
 * @brief OpenCL 数据类型表
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

const dtype_info_t* opencl_get_dtype_table(void);

static inline const dtype_info_t* opencl_dtype_info(ace_dtype_t dtype) {
    const dtype_info_t* table = opencl_get_dtype_table();
    if (dtype < 0 || dtype > ACE_DTYPE_BOOL) return &table[0];
    return &table[dtype];
}

#ifdef __cplusplus
}
#endif

#endif /* OPENCL_DTYPE_TABLE_H */
