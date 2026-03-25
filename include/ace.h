/**
 * @file ace.h
 * @brief AgierCompute - 跨平台GPU内核计算引擎
 * 
 * 用户唯一需要包含的头文件
 * 
 * 示例：
 *   ACE_KERNEL(vec_add,
 *       void vec_add(int n, T* a, T* b, T* c) {
 *           int i = GID;
 *           if (i < n) c[i] = a[i] + b[i];
 *       }
 *   );
 *   
 *   int n = N;
 *   void* args[] = {&n, buf_a, buf_b, buf_c};
 *   int types[] = {ACE_VAL, ACE_BUF, ACE_BUF, ACE_BUF};
 *   ace_kernel_invoke(dev, k_vec_add, ACE_DTYPE_FLOAT32, N, args, types, 4);
 */
#ifndef ACE_H
#define ACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* 版本 */
#define ACE_VERSION "1.0.0"

/* 平台导出宏 */
#ifdef _WIN32
    #ifdef ACE_CORE_EXPORTS
        #define ACE_API __declspec(dllexport)
    #else
        #define ACE_API __declspec(dllimport)
    #endif
#else
    #define ACE_API __attribute__((visibility("default")))
#endif

/* 错误码 */
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

/* 日志级别 */
typedef enum {
    ACE_LOG_ERROR = 0,
    ACE_LOG_WARN  = 1,
    ACE_LOG_INFO  = 2,
    ACE_LOG_DEBUG = 3,
} ace_log_level_t;

/* 设备类型 */
typedef enum {
    ACE_DEVICE_CPU    = 0,
    ACE_DEVICE_CUDA   = 1,
    ACE_DEVICE_OPENCL = 2,
    ACE_DEVICE_VULKAN = 3,
    ACE_DEVICE_METAL  = 4,
} ace_device_type_t;

/* 数据类型 */
typedef enum {
    ACE_DTYPE_FLOAT32 = 0,
    ACE_DTYPE_FLOAT64 = 1,
    ACE_DTYPE_INT32   = 2,
    ACE_DTYPE_INT64   = 3,
} ace_dtype_t;

/* 设备属性 */
typedef struct {
    ace_device_type_t type;
    char name[256];
    char vendor[128];
    size_t total_memory;
    size_t max_threads;
    int compute_units;
} ace_device_props_t;

/* 不透明句柄 */
typedef struct ace_device_* ace_device_t;
typedef struct ace_buffer_* ace_buffer_t;
typedef struct ace_stream_* ace_stream_t;
typedef struct ace_event_* ace_event_t;
typedef struct ace_mempool_* ace_mempool_t;
typedef void* ace_kernel_t;

/* 参数类型标记 */
#define ACE_VAL  0  /* 标量值（传指针） */
#define ACE_BUF  1  /* 缓冲区（传 ace_buffer_t） */

/* ============================================================================
 * 3D调度配置
 * ============================================================================ */

typedef struct {
    size_t x, y, z;
} ace_dim3_t;

typedef struct {
    ace_dim3_t grid;    /* 工作组数量 */
    ace_dim3_t block;   /* 每个工作组的线程数 */
} ace_launch_config_t;

/* 简化的1D调度配置宏 */
#define ACE_1D(n) ((ace_launch_config_t){ .grid = {n, 1, 1}, .block = {1, 1, 1} })
#define ACE_1D_BLOCK(n, b) ((ace_launch_config_t){ .grid = {(n+b-1)/b, 1, 1}, .block = {b, 1, 1} })

/* ============================================================================
 * 内存池配置
 * ============================================================================ */

typedef struct {
    size_t block_size;      /* 单个块大小 */
    size_t max_blocks;      /* 最大块数 */
    int thread_safe;        /* 是否线程安全 */
} ace_mempool_config_t;

#define ACE_MEMPOOL_DEFAULT ((ace_mempool_config_t){ .block_size = 4096, .max_blocks = 1024, .thread_safe = 1 })

/* ============================================================================
 * 内核定义宏 - 自动注册
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

#define GID        /* 全局线程ID */
#define LID        /* 局部线程ID */
#define BSIZE      /* 工作组大小 */
#define BARRIER()  /* 局部同步 */

/* ============================================================================
 * 公共 API
 * ============================================================================ */

/* 设备 */
ACE_API ace_error_t ace_device_count(ace_device_type_t type, int* count);
ACE_API ace_error_t ace_device_get(ace_device_type_t type, int idx, ace_device_t* dev);
ACE_API void ace_device_release(ace_device_t dev);
ACE_API ace_error_t ace_device_props(ace_device_t dev, ace_device_props_t* props);

/* 内存 */
ACE_API ace_error_t ace_buffer_alloc(ace_device_t dev, size_t size, ace_buffer_t* buf);
ACE_API void ace_buffer_free(ace_buffer_t buf);
ACE_API ace_error_t ace_buffer_write(ace_buffer_t buf, const void* data, size_t size);
ACE_API ace_error_t ace_buffer_read(ace_buffer_t buf, void* data, size_t size);

/* 同步 */
ACE_API ace_error_t ace_finish(ace_device_t dev);

/* ============================================================================
 * 流(Stream)管理 - 异步执行
 * ============================================================================ */

/* 创建/销毁流 */
ACE_API ace_error_t ace_stream_create(ace_device_t dev, ace_stream_t* stream);
ACE_API void ace_stream_destroy(ace_stream_t stream);

/* 流同步 */
ACE_API ace_error_t ace_stream_synchronize(ace_stream_t stream);
ACE_API ace_error_t ace_stream_wait_event(ace_stream_t stream, ace_event_t event);

/* 流属性 */
ACE_API ace_device_t ace_stream_get_device(ace_stream_t stream);

/* ============================================================================
 * 事件(Event)管理 - 精确同步
 * ============================================================================ */

/* 创建/销毁事件 */
ACE_API ace_error_t ace_event_create(ace_device_t dev, ace_event_t* event);
ACE_API void ace_event_destroy(ace_event_t event);

/* 记录和等待 */
ACE_API ace_error_t ace_event_record(ace_event_t event, ace_stream_t stream);
ACE_API ace_error_t ace_event_synchronize(ace_event_t event);

/* 计时 */
ACE_API ace_error_t ace_event_elapsed_time(ace_event_t start, ace_event_t end, float* ms);

/* 状态查询 */
ACE_API int ace_event_is_complete(ace_event_t event);

/* ============================================================================
 * 内存池管理
 * ============================================================================ */

/* 内存池配置 */
typedef struct {
    size_t initial_size;      /* 初始大小 */
    size_t max_size;         /* 最大大小 */
    size_t block_size;       /* 分配块大小 */
    int grow_on_demand;     /* 按需增长 */
} ace_mempool_config_t;

/* 创建/销毁内存池 */
ACE_API ace_error_t ace_mempool_create(ace_device_t dev, ace_mempool_config_t* config, ace_mempool_t* pool);
ACE_API ace_error_t ace_mempool_create_default(ace_device_t dev, ace_mempool_t* pool);
ACE_API void ace_mempool_destroy(ace_mempool_t pool);

/* 池化分配 */
ACE_API ace_buffer_t ace_mempool_alloc(ace_mempool_t pool, size_t size);
ACE_API void ace_mempool_free(ace_buffer_t buf);

/* 统计 */
ACE_API size_t ace_mempool_used(ace_mempool_t pool);
ACE_API size_t ace_mempool_available(ace_mempool_t pool);

/* 内核 */
ACE_API ace_kernel_t ace_register_kernel(const char* name, const char* src);

/* 内核调用 - 带流和3D调度 */
ACE_API ace_error_t ace_kernel_launch(ace_device_t dev, ace_kernel_t kernel,
                                       ace_dtype_t dtype, ace_launch_config_t* config,
                                       void** args, int* types, int nargs,
                                       ace_stream_t stream);

/* 兼容旧API */
ACE_API ace_error_t ace_kernel_invoke(ace_device_t dev, ace_kernel_t kernel,
                                       ace_dtype_t dtype, size_t n,
                                       void** args, int* types, int nargs);

/* ============================================================================
 * 辅助函数
 * ============================================================================ */

static inline const char* ace_dtype_name(ace_dtype_t dtype) {
    static const char* names[] = {"float", "double", "int", "long"};
    return names[dtype];
}

static inline size_t ace_dtype_size(ace_dtype_t dtype) {
    static const size_t sizes[] = {4, 8, 4, 8};
    return sizes[dtype];
}

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

/* 日志控制 */
ACE_API void ace_set_log_level(ace_log_level_t level);
ACE_API ace_log_level_t ace_get_log_level(void);

#ifdef __cplusplus
}
#endif

#endif /* ACE_H */