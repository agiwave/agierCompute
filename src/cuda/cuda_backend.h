/**
 * @file cuda_backend.h
 * @brief CUDA backend internal header
 */
#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include "ace.h"
#include "../ace_backend_api.h"
#include "cuda_dtype_table.h"

#ifdef CUDA_AVAILABLE

#include <cuda.h>

/* ============================================================================
 * Internal structures
 * ============================================================================ */

#define KERNEL_CACHE_SIZE 256

typedef struct cuda_kernel_s {
    int id;                /* 内核 ID */
    CUmodule module;
    CUfunction func;
    char* name;
    struct cuda_kernel_s* next;  /* 哈希表链式处理 */
} cuda_kernel_t;

typedef struct {
    cuda_kernel_t* buckets[KERNEL_CACHE_SIZE];
} cuda_kernel_cache_t;

typedef struct {
    CUdevice device;
    CUcontext context;
    char name[256];
    size_t total_mem;
    int sm_count;
    int max_threads;
    int compute_major;
    int compute_minor;
    cuda_kernel_cache_t kernel_cache;
    cuda_dtype_table_t dtype_table;  /* 设备级别的类型表 */
} cuda_device_t;

typedef struct {
    CUdeviceptr ptr;
    size_t size;
} cuda_buffer_t;

/* ============================================================================
 * Type utilities (cuda_type_utils.c)
 * ============================================================================ */

/* 设备级别的类型信息访问 */
static inline const char* cuda_get_type_name(const cuda_device_t* dev, ace_dtype_t dtype) {
    return cuda_dtype_info(&dev->dtype_table, dtype)->name;
}

char* cuda_translate_code(const cuda_device_t* dev, const char* name, const char* src, ace_dtype_t dtype);

/* ============================================================================
 * Device management (cuda_device.c)
 * ============================================================================ */

ace_error_t cuda_init(ace_backend_info_t* info);
void cuda_shutdown(ace_backend_info_t* info);
ace_error_t cuda_device_count(int* count);
ace_error_t cuda_device_get(int idx, void** dev);
void cuda_device_release(void* dev);
ace_error_t cuda_device_props(void* dev, void* props);

/* ============================================================================
 * Memory management (cuda_memory.c)
 * ============================================================================ */

ace_error_t cuda_mem_alloc(void* dev, size_t size, void** ptr);
void cuda_mem_free(void* dev, void* ptr);
ace_error_t cuda_mem_write(void* dev, void* dst, const void* src, size_t size);
ace_error_t cuda_mem_read(void* dev, void* dst, const void* src, size_t size);
ace_error_t cuda_finish(void* dev);

/* ============================================================================
 * Kernel management (cuda_kernel.c)
 * ============================================================================ */

ace_error_t cuda_kernel_launch(void* dev, ace_kernel_def_t* kernel_def,
                                ace_launch_config_t* cfg, void** args, size_t* sizes, int n);

#endif /* CUDA_AVAILABLE */

#endif /* CUDA_BACKEND_H */
