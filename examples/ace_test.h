/**
 * @file ace_test.h
 * @brief AgierCompute 标准测试框架
 * 
 * 提供：
 * - 设备遍历测试（使用 ACE_DEVICE_ALL）
 * - 矩阵测试（设备 x 测试用例）
 * - 性能基准测试
 */
#ifndef ACE_TEST_H
#define ACE_TEST_H

#include "ace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 测试结果
 * ============================================================================ */

typedef enum {
    ACE_TEST_SKIP = 0,   /* 跳过 */
    ACE_TEST_PASS = 1,   /* 通过 */
    ACE_TEST_FAIL = 2,   /* 失败 */
} ace_test_result_t;

/* ============================================================================
 * 测试用例定义
 * ============================================================================ */

typedef struct {
    const char* name;           /* 测试名称 */
    ace_test_result_t (*func)(ace_device_t dev, void* user_data);  /* 测试函数 */
    void* user_data;            /* 用户数据 */
} ace_test_case_t;

/* ============================================================================
 * 测试套件
 * ============================================================================ */

typedef struct {
    const char* name;           /* 套件名称 */
    ace_test_case_t* tests;     /* 测试用例数组 */
    int test_count;             /* 测试数量 */
    int passed;                 /* 通过数 */
    int failed;                 /* 失败数 */
    int skipped;                /* 跳过数 */
} ace_test_suite_t;

/* ============================================================================
 * 性能测试结果
 * ============================================================================ */

typedef struct {
    const char* test_name;
    const char* device_name;
    ace_device_type_t device_type;
    int device_index;
    double elapsed_ms;          /* 耗时（毫秒） */
    double gflops;              /* GFLOPS（如果适用） */
    double bandwidth_gbs;       /* 带宽 GB/s（如果适用） */
    int passed;                 /* 是否正确执行 */
} ace_benchmark_result_t;

/* ============================================================================
 * 测试框架 API
 * ============================================================================ */

/* 运行测试套件（遍历所有设备） */
ACE_API void ace_test_suite_run(ace_test_suite_t* suite);

/* 打印测试结果摘要 */
ACE_API void ace_test_print_summary(ace_test_suite_t* suite);

/* 保存基准测试结果到 CSV */
ACE_API void ace_benchmark_save_csv(ace_benchmark_result_t* results, 
                                     int count, const char* filename);

/* ============================================================================
 * 辅助宏
 * ============================================================================ */

/* 定义测试用例 */
#define ACE_TEST_DEFINE(name, func, data) \
    { name, func, data }

/* 断言辅助 */
#define ACE_TEST_ASSERT(cond) do { \
    if (!(cond)) return ACE_TEST_FAIL; \
} while(0)

/* 断言接近 */
#define ACE_TEST_ASSERT_NEAR(a, b, eps) do { \
    if (fabs((double)(a) - (double)(b)) > (eps)) return ACE_TEST_FAIL; \
} while(0)

/* 获取设备数量（所有类型） */
#define ACE_DEVICE_COUNT_ALL() ({ int c = 0; ace_device_count(ACE_DEVICE_ALL, &c); c; })

/* 获取设备（所有类型） */
#define ACE_DEVICE_GET_ALL(idx, dev) ace_device_get(ACE_DEVICE_ALL, idx, dev)

#ifdef __cplusplus
}
#endif

#endif /* ACE_TEST_H */
