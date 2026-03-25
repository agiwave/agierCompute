/**
 * @file ace.h
 * @brief AgierCompute - 跨平台GPU计算框架
 * 
 * 简洁的API设计，类似SYCL风格
 * 
 * 示例：
 *   // 1. 定义内核
 *   ACE_KERNEL(vec_add,
 *       void vec_add(int n, T* a, T* b, T* c) {
 *           int i = GID;
 *           if (i < n) c[i] = a[i] + b[i];
 *       }
 *   );
 *   
 *   // 2. 获取设备
 *   ace_device_t dev;
 *   ace_device_get(ACE_DEVICE_CPU, 0, &dev);
 *   
 *   // 3. 分配内存
 *   ace_buffer_t a, b, c;
 *   ace_buffer_alloc(dev, N * sizeof(float), &a);
 *   ace_buffer_alloc(dev, N * sizeof(float), &b);
 *   ace_buffer_alloc(dev, N * sizeof(float), &c);
 *   
 *   // 4. 写入数据
 *   ace_buffer_write(a, h_a, N * sizeof(float));
 *   ace_buffer_write(b, h_b, N * sizeof(float));
 *   
 *   // 5. 执行内核（自动异步）
 *   int n = N;
 *   void* args[] = {&n, a, b, c};
 *   int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
 *   ace_kernel_invoke(dev, k_vec_add, ACE_DTYPE_FLOAT32, N, args, types, 4);
 *   
 *   // 6. 读取结果（自动同步）
 *   ace_buffer_read(c, h_c, N * sizeof(float));
 *   
 *   // 7. 清理
 *   ace_buffer_free(a); ace_buffer_free(b); ace_buffer_free(c);
 *   ace_device_release(dev);
 */
#ifndef ACE_H
#define ACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

/* ============================================================================
 * 版本信息
 * ============================================================================ */

#define ACE_VERSION_MAJOR 1
#define ACE_VERSION_MINOR 0
#define ACE_VERSION_PATCH 0
#define ACE_VERSION "1.0.0"

/* ============================================================================
 * 平台导出宏
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
#define ACE_ERROR_IO       -6
#define ACE_ERROR_BACKEND  -7
#define ACE_ERROR_NOT_FOUND -8
#define ACE_ERROR_INVALID   -9

/* ============================================================================
 * 设备类型
 * ============================================================================ */

typedef enum {
    ACE_DEVICE_CPU    = 0,
    ACE_DEVICE_CUDA   = 1,
    ACE_DEVICE_OPENCL = 2,
    ACE_DEVICE_VULKAN = 3,
    ACE_DEVICE_METAL  = 4,
} ace_device_type_t;

/* ============================================================================
 * 数据类型
 * ============================================================================ */

typedef enum {
    ACE_DTYPE_FLOAT32 = 0,
    ACE_DTYPE_FLOAT64 = 1,
    ACE_DTYPE_INT32   = 2,
    ACE_DTYPE_INT64   = 3,
} ace_dtype_t;

/* ============================================================================
 * 设备属性
 * ============================================================================ */

typedef struct {
    ace_device_type_t type;
    char name[256];
    char vendor[128];
    size_t total_memory;
    size_t max_threads;
    int compute_units;
} ace_device_props_t;

/* ============================================================================
 * 不透明句柄
 * ============================================================================ */

typedef struct ace_device_* ace_device_t;
typedef struct ace_buffer_* ace_buffer_t;
typedef void* ace_kernel_t;

/* ============================================================================
 * 参数类型标记
 * ============================================================================ */

#define ACE_VAL  0  /* 标量值（传指针） */
#define ACE_BUF  1  /* 缓冲区（传 ace_buffer_t） */

/* ============================================================================
 * 3D调度配置
 * ============================================================================ */

struct ace_launch_config_ {
    size_t grid[3];     /* 工作组数量 */
    size_t block[3];    /* 每个工作组的线程数 */
    size_t shared_mem;  /* 动态共享内存大小（高级功能，默认0） */
};

typedef struct ace_launch_config_ ace_launch_config_t;

/* 调度配置辅助函数 */
static inline ace_launch_config_t ace_launch_1d(size_t n, size_t block) {
    ace_launch_config_t cfg = { 
        .grid = {(n + block - 1) / block, 1, 1}, 
        .block = {block, 1, 1},
        .shared_mem = 0
    };
    return cfg;
}

static inline ace_launch_config_t ace_launch_2d(size_t nx, size_t ny, size_t bx, size_t by) {
    ace_launch_config_t cfg = { 
        .grid = {(nx + bx - 1) / bx, (ny + by - 1) / by, 1},
        .block = {bx, by, 1},
        .shared_mem = 0
    };
    return cfg;
}

static inline ace_launch_config_t ace_launch_3d(size_t nx, size_t ny, size_t nz,
                                                  size_t bx, size_t by, size_t bz) {
    ace_launch_config_t cfg = { 
        .grid = {(nx + bx - 1) / bx, (ny + by - 1) / by, (nz + bz - 1) / bz},
        .block = {bx, by, bz},
        .shared_mem = 0
    };
    return cfg;
}

/* 简化宏 */
#define ACE_1D(n) ace_launch_1d(n, 256)
#define ACE_1D_BLOCK(n, b) ace_launch_1d(n, b)

/* ============================================================================
 * 内核定义宏
 * ============================================================================ */

#define ACE_KERNEL(name, code) \
    static ace_kernel_t k_##name = NULL; \
    static const char* _ace_src_##name = #code; \
    static ace_kernel_t _ace_get_##name(void) { \
        if (!k_##name) k_##name = ace_register_kernel(#name, _ace_src_##name); \
        return k_##name; \
    }

/* 简化的内核调用宏 */
#define ACE_CALL(dev, name, dtype, n, args, types, nargs) \
    ace_kernel_invoke(dev, _ace_get_##name(), ACE_DTYPE_##dtype, n, args, types, nargs)

/* ============================================================================
 * 内核语言内建变量
 * ============================================================================ */

#define GID        /* 全局线程ID - 在内核中使用 */
#define LID        /* 局部线程ID */
#define BSIZE      /* 工作组大小 */
#define BARRIER()  /* 局部同步 */

/* ============================================================================
 * 核心 API
 * ============================================================================ */

/* ----------------------------------------------------------------------------
 * 设备管理
 * ---------------------------------------------------------------------------- */

/* 获取指定类型的设备数量 */
ACE_API ace_error_t ace_device_count(ace_device_type_t type, int* count);

/* 获取设备 */
ACE_API ace_error_t ace_device_get(ace_device_type_t type, int idx, ace_device_t* dev);

/* 释放设备 */
ACE_API void ace_device_release(ace_device_t dev);

/* 获取设备属性 */
ACE_API ace_error_t ace_device_props(ace_device_t dev, ace_device_props_t* props);

/* ----------------------------------------------------------------------------
 * 内存管理（框架自动池化）
 * ---------------------------------------------------------------------------- */

/* 分配设备内存 */
ACE_API ace_error_t ace_buffer_alloc(ace_device_t dev, size_t size, ace_buffer_t* buf);

/* 释放设备内存 */
ACE_API void ace_buffer_free(ace_buffer_t buf);

/* 写入数据到设备（异步） */
ACE_API ace_error_t ace_buffer_write(ace_buffer_t buf, const void* data, size_t size);

/* 从设备读取数据（自动同步） */
ACE_API ace_error_t ace_buffer_read(ace_buffer_t buf, void* data, size_t size);

/* ----------------------------------------------------------------------------
 * 内核管理
 * ---------------------------------------------------------------------------- */

/* 注册内核 */
ACE_API ace_kernel_t ace_register_kernel(const char* name, const char* src);

/* 简化的内核调用 - 1D调度，自动异步 */
ACE_API ace_error_t ace_kernel_invoke(ace_device_t dev, ace_kernel_t kernel,
                                       ace_dtype_t dtype, size_t n,
                                       void** args, int* types, int nargs);

/* 高级内核调用 - 支持自定义3D调度 */
ACE_API ace_error_t ace_kernel_launch(ace_device_t dev, ace_kernel_t kernel,
                                       ace_dtype_t dtype, ace_launch_config_t* config,
                                       void** args, int* types, int nargs);

/* ----------------------------------------------------------------------------
 * 同步（可选，buffer_read会自动同步）
 * ---------------------------------------------------------------------------- */

/* 等待设备上所有操作完成 */
ACE_API ace_error_t ace_finish(ace_device_t dev);

/* ace_finish的别名 */
static inline ace_error_t ace_sync(ace_device_t dev) { return ace_finish(dev); }

/* ============================================================================
 * 辅助函数
 * ============================================================================ */

/* 获取数据类型名称 */
static inline const char* ace_dtype_name(ace_dtype_t dtype) {
    static const char* names[] = {"float", "double", "int", "long"};
    return names[dtype];
}

/* 获取数据类型大小 */
static inline size_t ace_dtype_size(ace_dtype_t dtype) {
    static const size_t sizes[] = {4, 8, 4, 8};
    return sizes[dtype];
}

/* 获取错误描述 */
static inline const char* ace_error_string(ace_error_t err) {
    switch (err) {
        case ACE_OK:              return "OK";
        case ACE_ERROR:           return "General error";
        case ACE_ERROR_MEM:       return "Memory error";
        case ACE_ERROR_DEVICE:    return "Device error";
        case ACE_ERROR_COMPILE:   return "Compile error";
        case ACE_ERROR_LAUNCH:    return "Launch error";
        case ACE_ERROR_IO:        return "I/O error";
        case ACE_ERROR_BACKEND:   return "Backend error";
        case ACE_ERROR_NOT_FOUND: return "Not found";
        case ACE_ERROR_INVALID:   return "Invalid argument";
        default:                  return "Unknown error";
    }
}

/* ============================================================================
 * 多设备管理 API - 跨 GPU 运行
 * ============================================================================ */

/* 设备列表 */
typedef struct {
    ace_device_t* devices;
    int count;
    ace_device_type_t type;
} ace_device_list_t;

/* 获取所有可用设备 */
ACE_API ace_error_t ace_device_get_all(ace_device_list_t* list);

/* 释放设备列表 */
ACE_API void ace_device_list_release(ace_device_list_t* list);

/* 选择最佳设备（优先 GPU，其次 CPU） */
ACE_API ace_error_t ace_device_select_best(ace_device_t* dev);

/* ============================================================================
 * 数据并行 API - 自动跨设备分片
 * ============================================================================ */

/* 分片缓冲区 */
typedef struct {
    ace_buffer_t* buffers;
    size_t* offsets;
    size_t* sizes;
    int count;
} ace_sharded_buffer_t;

/* 创建分片缓冲区 - 自动在多个设备上分配 */
ACE_API ace_error_t ace_buffer_alloc_sharded(
    ace_device_list_t* devices,
    size_t total_size,
    ace_sharded_buffer_t* sharded
);

/* 释放分片缓冲区 */
ACE_API void ace_buffer_free_sharded(ace_sharded_buffer_t* sharded);

/* 写入分片缓冲区 */
ACE_API ace_error_t ace_buffer_write_sharded(
    ace_sharded_buffer_t* sharded,
    const void* data,
    size_t total_size
);

/* 读取分片缓冲区 */
ACE_API ace_error_t ace_buffer_read_sharded(
    ace_sharded_buffer_t* sharded,
    void* data,
    size_t total_size
);

/* 跨设备内核执行 - 自动分片并行 */
ACE_API ace_error_t ace_kernel_invoke_sharded(
    ace_device_list_t* devices,
    ace_kernel_t kernel,
    ace_dtype_t dtype,
    size_t n,
    void** args,
    int* types,
    int nargs
);

/* 等待所有设备完成 */
ACE_API ace_error_t ace_finish_all(ace_device_list_t* devices);

/* ============================================================================
 * Stream API - 异步执行（简化版）
 * ============================================================================ */

typedef struct ace_stream_* ace_stream_t;

/* 创建流 */
ACE_API ace_error_t ace_stream_create(ace_device_t dev, ace_stream_t* stream);

/* 销毁流 */
ACE_API void ace_stream_destroy(ace_stream_t stream);

/* 在流上执行内核（异步，不阻塞） */
ACE_API ace_error_t ace_stream_launch(
    ace_stream_t stream,
    ace_kernel_t kernel,
    ace_dtype_t dtype,
    size_t n,
    void** args,
    int* types,
    int nargs
);

/* 异步内存传输 */
ACE_API ace_error_t ace_stream_memcpy_h2d(ace_stream_t stream, ace_buffer_t dst, const void* src, size_t size);
ACE_API ace_error_t ace_stream_memcpy_d2h(ace_stream_t stream, void* dst, ace_buffer_t src, size_t size);

/* 流同步（等待流完成） */
ACE_API ace_error_t ace_stream_synchronize(ace_stream_t stream);

/* 获取默认流（同步） */
ACE_API ace_stream_t ace_stream_default(ace_device_t dev);

/* ============================================================================
 * 内存池 API - 简化版
 * ============================================================================ */

typedef struct ace_mempool_* ace_mempool_t;

/* 创建内存池 */
ACE_API ace_mempool_t ace_mempool_create(ace_device_t dev);

/* 销毁内存池 */
ACE_API void ace_mempool_destroy(ace_mempool_t pool);

/* 从内存池分配 */
ACE_API ace_error_t ace_mempool_alloc(ace_mempool_t pool, size_t size, ace_buffer_t* buf);

/* 释放回内存池 */
ACE_API void ace_mempool_free(ace_mempool_t pool, ace_buffer_t buf);

/* ============================================================================
 * 实用计算原语
 * ============================================================================ */

/* 向量加法：y = a + b */
ACE_API ace_error_t ace_vec_add(ace_stream_t stream, int n, ace_buffer_t a, ace_buffer_t b, ace_buffer_t y);

/* 向量缩放：y = alpha * x */
ACE_API ace_error_t ace_vec_scale(ace_stream_t stream, int n, float alpha, ace_buffer_t x, ace_buffer_t y);

/* 向量点积：result = dot(x, y) */
ACE_API ace_error_t ace_vec_dot(ace_stream_t stream, int n, ace_buffer_t x, ace_buffer_t y, ace_buffer_t result);

/* 矩阵乘法：C = A * B (A:mxk, B:kxn, C:mxn) */
ACE_API ace_error_t ace_matmul(
    ace_stream_t stream,
    int m, int n, int k,
    ace_buffer_t A, ace_buffer_t B, ace_buffer_t C
);

/* ReLU 激活：y = max(0, x) */
ACE_API ace_error_t ace_relu(ace_stream_t stream, int n, ace_buffer_t x, ace_buffer_t y);

/* Sigmoid 激活：y = 1 / (1 + exp(-x)) */
ACE_API ace_error_t ace_sigmoid(ace_stream_t stream, int n, ace_buffer_t x, ace_buffer_t y);

/* ============================================================================
 * 辅助函数 - 多设备
 * ============================================================================ */

/* 打印设备信息 */
static inline void ace_device_print_info(ace_device_t dev) {
    ace_device_props_t props;
    if (ace_device_props(dev, &props) == ACE_OK) {
        const char* type_names[] = {"CPU", "CUDA", "OpenCL", "Vulkan", "Metal"};
        const char* t = (props.type >= 0 && props.type <= 4) ? type_names[props.type] : "Unknown";
        printf("Device: %s (%s)\n", props.name, t);
        printf("  Vendor: %s\n", props.vendor);
        printf("  Max threads: %zu\n", props.max_threads);
        printf("  Compute units: %d\n", props.compute_units);
        if (props.total_memory > 0) {
            printf("  Total memory: %zu MB\n", props.total_memory / (1024 * 1024));
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* ACE_H */
