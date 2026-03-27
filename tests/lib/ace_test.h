/**
 * @file ace_test.h
 * @brief AgierCompute 测试框架头文件
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

typedef enum {
    ACE_TEST_SKIP = 0,
    ACE_TEST_PASS = 1,
    ACE_TEST_FAIL = 2,
} ace_test_result_t;

typedef struct {
    const char* name;
    ace_test_result_t (*func)(ace_device_t dev, void* user_data);
    void* user_data;
} ace_test_case_t;

typedef struct {
    const char* name;
    ace_test_case_t* tests;
    int test_count;
    int passed, failed, skipped;
} ace_test_suite_t;

typedef struct {
    const char* test_name;
    const char* device_name;
    ace_device_type_t device_type;
    int device_index;
    double elapsed_ms, gflops, bandwidth_gbs;
    int passed;
} ace_benchmark_result_t;

/* API */
void ace_test_suite_run(ace_test_suite_t* suite);
void ace_test_print_summary(ace_test_suite_t* suite);
void ace_benchmark_save_csv(ace_benchmark_result_t* results, int count, const char* filename);

/* 辅助宏 */
#define ACE_TEST_DEFINE(name, func, data) { name, func, data }
#define ACE_TEST_ASSERT(cond) do { if (!(cond)) return ACE_TEST_FAIL; } while(0)
#define ACE_TEST_ASSERT_NEAR(a, b, eps) do { if (fabs((double)(a) - (double)(b)) > (eps)) return ACE_TEST_FAIL; } while(0)

#ifdef __cplusplus
}
#endif

#endif /* ACE_TEST_H */
