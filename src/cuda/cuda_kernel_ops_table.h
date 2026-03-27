/**
 * @file cuda_kernel_ops_table.h
 * @brief CUDA 内核操作表
 */
#ifndef CUDA_KERNEL_OPS_TABLE_H
#define CUDA_KERNEL_OPS_TABLE_H

#include "cuda_dtype_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*op_inject_fn)(char* buf, const dtype_info_t* type);

typedef struct {
    const char* name;
    op_inject_fn inject;
} kernel_op_t;

const kernel_op_t* cuda_get_kernel_ops_table(void);
const kernel_op_t* cuda_find_kernel_op(const char* name);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_KERNEL_OPS_TABLE_H */
