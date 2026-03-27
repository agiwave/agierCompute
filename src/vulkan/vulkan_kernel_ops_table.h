/**
 * @file vulkan_kernel_ops_table.h
 * @brief Vulkan 内核操作表
 *
 * 设计理念:
 * - 框架只负责查表和调用 inject 函数
 * - inject 函数接收类型信息，自己决定如何生成代码
 * - 框架不知道原生/模拟，不知道具体类型，不知道具体操作
 */
#ifndef VULKAN_KERNEL_OPS_TABLE_H
#define VULKAN_KERNEL_OPS_TABLE_H

#include "vulkan_dtype_table.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 内核操作注入函数
 *
 * @param buf 输出缓冲区
 * @param type 类型信息 (函数内部根据 type->name 等决定如何生成代码)
 * @return 写入的字符数
 */
typedef int (*op_inject_fn)(char* buf, const dtype_info_t* type);

/**
 * @brief 内核操作信息
 */
typedef struct {
    const char* name;           /* 操作名称 (如 "kadd", "ksub") */
    op_inject_fn inject;        /* 注入函数 (唯一入口，内部处理所有逻辑) */
} kernel_op_t;

/**
 * @brief 获取内核操作表
 */
// const kernel_op_t* get_kernel_ops_table(void);

/**
 * @brief 根据名称查找内核操作
 */
const kernel_op_t* find_kernel_op(const char* name);

#ifdef __cplusplus
}
#endif

#endif /* VULKAN_KERNEL_OPS_TABLE_H */
