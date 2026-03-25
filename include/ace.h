/**
 * @file ace.h
 * @brief AgierCompute - 跨平台 GPU 计算框架
 *
 * CUDA 风格的极简 API
 *
 * 对比 CUDA:
 *   CUDA                          AgierCompute
 *   cudaSetDevice()           ->  ace_set_device()
 *   cudaMalloc(&d, size)      ->  ace_malloc(&d, size)
 *   cudaFree(d)               ->  ace_free(d)
 *   cudaMemcpy(d, h, size)    ->  ace_memcpy(d, h, size)
 *   kernel<<<grid,block>>>(..) ->  ace_launch(kernel, n, ...)
 *   cudaDeviceSynchronize()   ->  ace_sync()
 *
 * 示例:
 *   // 1. 定义内核
 *   ACE_KERNEL(vec_add,
 *       void(int n, float* a, float* b, float* c) {
 *           int i = GID;
 *           if (i < n) c[i] = a[i] + b[i];
 *       }
 *   );
 *
 *   // 2. 选择设备
 *   ace_set_device(ACE_DEVICE_CPU);
 *
 *   // 3. 分配内存
 *   float *d_a, *d_b, *d_c;
 *   ace_malloc(&d_a, N * sizeof(float));
 *   ace_malloc(&d_b, N * sizeof(float));
 *   ace_malloc(&d_c, N * sizeof(float));
 *
 *   // 4. 拷贝数据
 *   ace_memcpy(d_a, h_a, N * sizeof(float));
 *   ace_memcpy(d_b, h_b, N * sizeof(float));
 *
 *   // 5. 启动内核
 *   ace_launch(vec_add, N, "iffff", N, d_a, d_b, d_c);
 *
 *   // 6. 同步并读取
 *   ace_sync();
 *   ace_memcpy(h_c, d_c, N * sizeof(float));
 *
 *   // 7. 释放
 *   ace_free(d_a); ace_free(d_b); ace_free(d_c);
 */
#ifndef ACE_H
#define ACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>

/* ============================================================================
 * 版本
 * ============================================================================ */
#define ACE_VERSION "1.0.0"

/* ============================================================================
 * 导出宏
 * ============================================================================ */
#ifdef _WIN32
    #ifdef ACE_CORE_EXPORTS
        #define ACE_API __declspec(dllexport)
    #else
        #define ACE_API __declspec(dllimport)
    #endif
#else
    #define ACE_API __attribute__((visibility("default")))
#endif

/* ============================================================================
 * 错误码
 * ============================================================================ */
typedef int ace_error_t;
#define ACE_OK              0
#define ACE_ERROR          -1
#define ACE_ERROR_MEM      -2
#define ACE_ERROR_DEVICE   -3
#define ACE_ERROR_COMPILE  -4
#define ACE_ERROR_LAUNCH   -5

/* ============================================================================
 * 设备类型
 * ============================================================================ */
typedef enum {
    ACE_DEVICE_CPU    = 0,   /* CPU (多线程) */
    ACE_DEVICE_CUDA   = 1,   /* NVIDIA GPU */
    ACE_DEVICE_OPENCL = 2,   /* OpenCL 设备 */
    ACE_DEVICE_VULKAN = 3,   /* Vulkan 计算 */
} ace_device_type_t;

/* ============================================================================
 * 数据类型
 * ============================================================================ */
typedef enum {
    ACE_FLOAT32 = 0,
    ACE_FLOAT64 = 1,
    ACE_INT32   = 2,
    ACE_INT64   = 3,
} ace_dtype_t;

/* ============================================================================
 * 设备信息
 * ============================================================================ */
typedef struct {
    ace_device_type_t type;
    char name[256];
    int compute_units;
    size_t memory;
} ace_device_info_t;

/* ============================================================================
 * 核心 API - CUDA 风格
 * ============================================================================ */

/* --- 设备管理 --- */

/* 设置当前设备 */
ACE_API ace_error_t ace_set_device(ace_device_type_t type, int index);

/* 获取当前设备 */
ACE_API ace_device_type_t ace_get_device(void);

/* 获取设备信息 */
ACE_API ace_error_t ace_get_device_info(ace_device_info_t* info);

/* 同步设备 */
ACE_API ace_error_t ace_sync(void);

/* 打印设备信息 */
ACE_API void ace_print_device(void);

/* --- 内存管理 --- */

/* 分配设备内存 */
ACE_API ace_error_t ace_malloc(void** ptr, size_t size);

/* 释放设备内存 */
ACE_API ace_error_t ace_free(void* ptr);

/* 主机->设备拷贝 */
ACE_API ace_error_t ace_memcpy_d2d(void* dst, const void* src, size_t size);

/* 主机->设备拷贝 */
ACE_API ace_error_t ace_memcpy_h2d(void* dst, const void* src, size_t size);

/* 设备->主机拷贝 */
ACE_API ace_error_t ace_memcpy_d2h(void* dst, const void* src, size_t size);

/* 便捷拷贝（自动判断方向） */
ACE_API ace_error_t ace_memcpy(void* dst, const void* src, size_t size);

/* --- 内核执行 --- */

/* 内核句柄 */
typedef void* ace_kernel_t;

/* 注册内核 */
ACE_API ace_kernel_t ace_kernel_register(const char* name, const char* src);

/* 启动内核 (1D) */
ACE_API ace_error_t ace_launch(
    ace_kernel_t kernel,
    size_t global_size,
    const char* signature,  /* "i"=int, "f"=float, "d"=double, "l"=long, "p"=pointer */
    ...
);

/* 启动内核 (3D) */
ACE_API ace_error_t ace_launch_3d(
    ace_kernel_t kernel,
    size_t grid_x, size_t grid_y, size_t grid_z,
    size_t block_x, size_t block_y, size_t block_z,
    const char* signature,
    ...
);

/* ============================================================================
 * 内核定义宏
 * ============================================================================ */

/* 定义内核（文件作用域） */
#define ACE_KERNEL(name, code) \
    static const char* _ace_src_##name = #code; \
    static ace_kernel_t _ace_kern_##name = NULL; \
    static ace_kernel_t ace_kernel_##name(void) { \
        if (!_ace_kern_##name) \
            _ace_kern_##name = ace_kernel_register(#name, _ace_src_##name); \
        return _ace_kern_##name; \
    }

/* 便捷启动宏 */
#define ACE_LAUNCH(name, n, sig, ...) \
    ace_launch(ace_kernel_##name(), n, sig, ##__VA_ARGS__)

/* ============================================================================
 * 内建变量（内核中使用）
 * ============================================================================ */
/*
 * GID      - 全局线程 ID (类似 CUDA threadIdx + blockIdx)
 * LID      - 局部线程 ID
 * BSIZE    - 块大小
 */

/* ============================================================================
 * 辅助函数
 * ============================================================================ */
ACE_API const char* ace_strerror(ace_error_t err);
ACE_API const char* ace_dtype_str(ace_dtype_t dtype);

/* 获取错误描述 */
static inline const char* ace_error_string(ace_error_t err) {
    switch (err) {
        case ACE_OK: return "OK";
        case ACE_ERROR: return "Error";
        case ACE_ERROR_MEM: return "Memory error";
        case ACE_ERROR_DEVICE: return "Device error";
        case ACE_ERROR_COMPILE: return "Compile error";
        case ACE_ERROR_LAUNCH: return "Launch error";
        default: return "Unknown";
    }
}

#ifdef __cplusplus
}
#endif

#endif /* ACE_H */
