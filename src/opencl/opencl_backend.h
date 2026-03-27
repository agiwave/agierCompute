/**
 * @file opencl_backend.h
 * @brief OpenCL backend internal header
 */
#ifndef OPENCL_BACKEND_H
#define OPENCL_BACKEND_H

#include "ace.h"
#include "../ace_backend_api.h"

#ifdef OPENCL_AVAILABLE

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

/* ============================================================================
 * Internal structures
 * ============================================================================ */

#define KERNEL_CACHE_SIZE 256

typedef struct ocl_kernel_s {
    int id;                /* 内核 ID */
    cl_kernel kernel;
    cl_program program;
    char* name;
    struct ocl_kernel_s* next;
} ocl_kernel_t;

typedef struct {
    ocl_kernel_t* buckets[KERNEL_CACHE_SIZE];
} ocl_kernel_cache_t;

typedef struct {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    char name[256];
    size_t total_mem;
    int compute_units;
    int max_threads;
    ocl_kernel_cache_t kernel_cache;
} ocl_device_t;

typedef struct {
    cl_mem mem;
    size_t size;
} ocl_buffer_t;

/* Global platform */
extern cl_platform_id g_opencl_platform;

/* Device extensions status */
typedef struct {
    int has_fp16;              /* cl_khr_fp16 */
    int has_fp64;              /* cl_khr_fp64 */
    int has_int64;             /* native support */
    int has_int8;              /* cl_khr_int8 */
    int has_int16;             /* cl_khr_int16 */
    int has_8bit_storage;      /* cl_khr_8bit_storage */
    int has_16bit_storage;     /* cl_khr_16bit_storage */
} ocl_device_extensions_t;

extern ocl_device_extensions_t g_device_exts;

/* ============================================================================
 * Type utilities (opencl_type_utils.c)
 * ============================================================================ */

const char* ocl_get_type_name(ace_dtype_t dtype);
const char* ocl_get_extension(ace_dtype_t dtype);
char* ocl_translate_code(const char* name, const char* src, ace_dtype_t dtype);

/* ============================================================================
 * Device management (opencl_device.c)
 * ============================================================================ */

ace_error_t ocl_init(ace_backend_info_t* info);
void ocl_shutdown(ace_backend_info_t* info);
ace_error_t ocl_device_count(int* count);
ace_error_t ocl_device_get(int idx, void** dev);
void ocl_device_release(void* dev);
ace_error_t ocl_device_props(void* dev, void* props);

/* ============================================================================
 * Memory management (opencl_memory.c)
 * ============================================================================ */

ace_error_t ocl_mem_alloc(void* dev, size_t size, void** ptr);
void ocl_mem_free(void* dev, void* ptr);
ace_error_t ocl_mem_write(void* dev, void* dst, const void* src, size_t size);
ace_error_t ocl_mem_read(void* dev, void* dst, const void* src, size_t size);
ace_error_t ocl_finish(void* dev);

/* ============================================================================
 * Kernel management (opencl_kernel.c)
 * ============================================================================ */

ace_error_t ocl_kernel_launch(void* dev, ace_kernel_def_t* kernel_def,
                               ace_launch_config_t* cfg, void** args, size_t* sizes, int n);

#endif /* OPENCL_AVAILABLE */

#endif /* OPENCL_BACKEND_H */
